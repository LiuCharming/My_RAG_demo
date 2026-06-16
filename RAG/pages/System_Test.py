from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

import streamlit as st
from datasets import load_dataset
from rag_pipeline import get_llm

from rag_service import ask_rag, warmup_runtime
from rag_settings import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_SOURCE_URL,
    RAGSettings,
    UPLOADS_DIR,
    sanitize_name,
)
from ui_helpers import get_collection_name, list_custom_knowledge_bases


def build_source_config():
    source_type = st.selectbox(
        "Knowledge source",
        options=["web", "dataset", "custom"],
        index=1,
    )
    source_url = DEFAULT_SOURCE_URL
    custom_source_name = None
    dataset_name = DEFAULT_DATASET_NAME
    dataset_split = DEFAULT_DATASET_SPLIT

    if source_type == "web":
        source_url = st.text_input("Web article URL", value=DEFAULT_SOURCE_URL)
    elif source_type == "dataset":
        dataset_name = st.text_input("Dataset name", value=DEFAULT_DATASET_NAME)
        dataset_split = st.text_input("Dataset split", value=DEFAULT_DATASET_SPLIT)
    else:
        existing_custom_bases = list_custom_knowledge_bases()
        if existing_custom_bases:
            custom_source_name = st.selectbox(
                "Custom knowledge base",
                options=existing_custom_bases,
            )
        else:
            custom_source_name = st.text_input(
                "Custom knowledge base",
                value="python_manual",
            )
            st.info("No custom knowledge base found yet. Build one from the main page first.")

    collection_name = get_collection_name(
        source_type=source_type,
        custom_source_name=custom_source_name,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    uploads_target = UPLOADS_DIR / sanitize_name(collection_name)
    uploaded_files_dir = str(uploads_target) if uploads_target.exists() else None
    return {
        "source_type": source_type,
        "source_url": source_url,
        "custom_source_name": custom_source_name,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "collection_name": collection_name,
        "uploaded_files_dir": uploaded_files_dir,
    }


def parse_history_lines(raw_text: str) -> list[dict]:
    messages = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        role_text, content = line.split(":", 1)
        role = role_text.strip().lower()
        content = content.strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def parse_batch_cases(raw_text: str) -> list[dict]:
    cases = []
    for index, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) == 1:
            question = parts[0]
            expected = ""
        else:
            question = parts[0]
            expected = parts[1]
        if not question:
            continue
        cases.append(
            {
                "case_id": index,
                "question": question,
                "expected": expected,
            }
        )
    return cases


@st.cache_data(show_spinner=False)
def load_dataset_samples(dataset_name: str, split: str, limit: int) -> list[dict]:
    dataset = load_dataset(dataset_name, split=split)
    rows = []
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        answers = row.get("answers", {}) or {}
        answer_texts = [text.strip() for text in answers.get("text", []) if str(text).strip()]
        rows.append(
            {
                "id": row.get("id", f"{split}_{index}"),
                "question": row.get("question", ""),
                "context": row.get("context", ""),
                "answers": answer_texts,
            }
        )
    return rows


def answer_contains_reference(answer: str, references: list[str]) -> bool:
    normalized_answer = (answer or "").strip()
    if not normalized_answer or not references:
        return False
    return any(reference and reference in normalized_answer for reference in references)


def top_chunk_contains_reference(retrieved_docs: list, references: list[str]) -> bool:
    if not retrieved_docs or not references:
        return False
    top_chunk_text = getattr(retrieved_docs[0], "page_content", "") or ""
    return any(reference and reference in top_chunk_text for reference in references)


def normalize_text(text: str) -> str:
    return "".join(str(text or "").strip().lower().split())


def exact_match_any(prediction: str, references: list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    for reference in references:
        if normalized_prediction == normalize_text(reference):
            return 1.0
    return 0.0


def char_f1(prediction: str, reference: str) -> float:
    pred_chars = list(normalize_text(prediction))
    ref_chars = list(normalize_text(reference))
    if not pred_chars or not ref_chars:
        return 0.0

    ref_counts = {}
    for char in ref_chars:
        ref_counts[char] = ref_counts.get(char, 0) + 1

    overlap = 0
    for char in pred_chars:
        if ref_counts.get(char, 0) > 0:
            overlap += 1
            ref_counts[char] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_chars)
    recall = overlap / len(ref_chars)
    return 2 * precision * recall / (precision + recall)


def best_f1_against_references(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(char_f1(prediction, reference) for reference in references)


def doc_contains_reference(doc, references: list[str]) -> bool:
    content = getattr(doc, "page_content", "") or ""
    return any(reference and reference in content for reference in references)


def retrieval_metrics_at_k(retrieved_docs: list, references: list[str], k: int) -> dict:
    top_docs = retrieved_docs[:k]
    hits = [doc_contains_reference(doc, references) for doc in top_docs]
    hit_count = sum(hits)
    first_hit_rank = None
    for index, hit in enumerate(hits, start=1):
        if hit:
            first_hit_rank = index
            break
    return {
        "top1_hit": 1.0 if hits[:1] and hits[0] else 0.0,
        "topk_hit": 1.0 if hit_count > 0 else 0.0,
        "recall_at_k": 1.0 if hit_count > 0 else 0.0,
        "precision_at_k": (hit_count / len(top_docs)) if top_docs else 0.0,
        "mrr": (1.0 / first_hit_rank) if first_hit_rank else 0.0,
        "first_hit_rank": first_hit_rank,
    }


def safe_parse_binary_score(text: str) -> float:
    normalized = str(text or "").strip()
    if normalized in {"1", "1.0"}:
        return 1.0
    if normalized in {"0", "0.0"}:
        return 0.0
    for char in normalized:
        if char == "1":
            return 1.0
        if char == "0":
            return 0.0
    return 0.0


def judge_with_llm(prompt: str, model_name: str) -> float:
    llm = get_llm(model_name)
    response = llm.invoke(prompt).content
    return safe_parse_binary_score(response)


def judge_answer_relevance(question: str, prediction: str, model_name: str) -> float:
    prompt = f"""
You are evaluating whether a predicted answer is relevant to the user's question.

Question:
{question}

Predicted answer:
{prediction}

Return:
- 1 if the answer directly addresses the question and is relevant.
- 0 if the answer is off-topic, evasive, or does not actually answer the question.

Only output 0 or 1.
""".strip()
    return judge_with_llm(prompt, model_name)


def judge_answer_faithfulness(
    question: str,
    prediction: str,
    retrieved_docs: list,
    model_name: str,
) -> float:
    context = "\n\n".join(
        getattr(doc, "page_content", "") for doc in retrieved_docs[:3]
    )
    prompt = f"""
You are evaluating whether a RAG answer is faithful to the retrieved context.

Question:
{question}

Retrieved context:
{context}

Predicted answer:
{prediction}

Return:
- 1 if the answer is supported by the retrieved context.
- 0 if the answer contains unsupported claims, hallucinations, or information not grounded in the context.

Only output 0 or 1.
""".strip()
    return judge_with_llm(prompt, model_name)


def evaluate_dataset_sample(sample: dict) -> dict:
    result = run_case(sample["question"])
    answer_hit = answer_contains_reference(result["answer"], sample["answers"])
    answer_em = exact_match_any(result["answer"], sample["answers"])
    answer_f1 = best_f1_against_references(result["answer"], sample["answers"])
    retrieval_metrics = retrieval_metrics_at_k(
        result["retrieved_docs"],
        sample["answers"],
        k=max(rerank_k if use_rerank else retrieve_k, 1),
    )
    relevance_score = None
    faithfulness_score = None
    if enable_llm_judge:
        relevance_score = judge_answer_relevance(
            sample["question"],
            result["answer"],
            settings_defaults.chat_model,
        )
        faithfulness_score = judge_answer_faithfulness(
            sample["question"],
            result["answer"],
            result["retrieved_docs"],
            settings_defaults.chat_model,
        )
    rewrite_used = bool(result.get("rewrite_used", False))
    return {
        "sample": sample,
        "result": result,
        "answer_hit": answer_hit,
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "retrieval_metrics": retrieval_metrics,
        "relevance_score": relevance_score,
        "faithfulness_score": faithfulness_score,
        "rewrite_used": rewrite_used,
    }


st.set_page_config(page_title="System Test", page_icon="T", layout="wide")

if "runtime_warmed_up" not in st.session_state:
    with st.spinner("Warming up models..."):
        warmup_runtime()
    st.session_state.runtime_warmed_up = True

st.title("System Test")
st.caption("Use this page to validate retrieval, rewrite, rerank, and answer quality.")
st.page_link("app.py", label="Back to RAG chat", icon=":material/chat:")
st.page_link(
    "pages/Knowledge_Base_Preview.py",
    label="Open knowledge base preview",
    icon=":material/search:",
)

with st.sidebar:
    settings_defaults = RAGSettings()
    st.subheader("Test Scope")
    source_config = build_source_config()

    st.subheader("Retrieval Settings")
    answer_model_labels = {
        "deepseek_api": f"DeepSeek API ({settings_defaults.chat_model})",
        "local_qwen": f"Local Qwen ({settings_defaults.local_chat_model})",
        "vllm_openai": f"vLLM Local Server ({settings_defaults.vllm_model})",
    }
    chat_backend = st.selectbox(
        "Answer model",
        options=["deepseek_api", "local_qwen", "vllm_openai"],
        index=0,
        format_func=lambda value: answer_model_labels.get(value, value),
    )
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        options=["vector", "bm25", "hybrid"],
        index=2,
    )
    retrieve_k = st.slider("Retrieve k", min_value=2, max_value=20, value=10, step=1)
    use_rerank = st.checkbox("Use rerank", value=True)
    rerank_k = st.slider("Rerank k", min_value=1, max_value=10, value=2, step=1)

    st.subheader("Chunk Settings")
    chunk_size = st.slider("Chunk size", min_value=300, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=500, value=200, step=50)

left, right = st.columns([1.2, 1], gap="large")

with left:
    st.subheader("Single Test")
    question = st.text_input("Question", value="大船是什么？")
    history_input = st.text_area(
        "Conversation history (optional)",
        value="user: 舵\nassistant: 舵是航行设备上用于改变或保持航行方向的一种装置。\nuser: 它有什么用呢？",
        height=120,
        help="One message per line. Format: user: ... or assistant: ...",
    )
    run_single = st.button("Run single test", use_container_width=True)

    st.subheader("Batch Test")
    batch_input = st.text_area(
        "Batch cases",
        value=(
            "舵是什么？\t应该回答舵的定义\n"
            "它有什么用呢？\t应该结合历史改写\n"
            "大船是什么？\t应该避免被错误历史带偏"
        ),
        height=220,
        help="Use one case per line. Recommended format: question<TAB>expected behavior",
    )
    batch_history_mode = st.selectbox(
        "Batch history mode",
        options=["No shared history", "Carry previous user question"],
        index=0,
        help="Carry previous user question is useful for testing simple multi-turn behavior.",
    )
    run_batch = st.button("Run batch test", use_container_width=True)

    st.subheader("Dataset Evaluation")
    eval_split = st.selectbox(
        "Evaluation split",
        options=["train", "validation", "test"],
        index=0,
        disabled=source_config["source_type"] != "dataset",
    )
    eval_sample_size = st.slider(
        "Sample size",
        min_value=5,
        max_value=200,
        value=20,
        step=5,
        disabled=source_config["source_type"] != "dataset",
    )
    eval_execution_mode = st.selectbox(
        "Execution mode",
        options=["serial", "parallel"],
        index=0,
        disabled=source_config["source_type"] != "dataset",
        help="Parallel mode is useful for throughput testing. Local Qwen is best kept at low concurrency.",
    )
    requested_concurrency = st.selectbox(
        "Concurrency",
        options=[1, 2, 4, 8],
        index=0,
        disabled=source_config["source_type"] != "dataset",
    )
    run_eval = st.button(
        "Run dataset evaluation",
        use_container_width=True,
        disabled=source_config["source_type"] != "dataset",
    )
    enable_llm_judge = st.checkbox(
        "Enable LLM judge metrics",
        value=False,
        help="Adds relevance and faithfulness scoring. This is slower and may use API calls.",
    )

with right:
    st.subheader("Active Config")
    st.json(
        {
            **source_config,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_mode": retrieval_mode,
            "retrieve_k": retrieve_k,
            "use_rerank": use_rerank,
            "rerank_k": rerank_k,
            "chat_backend": chat_backend,
            "chat_model": (
                settings_defaults.local_chat_model
                if chat_backend == "local_qwen"
                else settings_defaults.vllm_model
                if chat_backend == "vllm_openai"
                else settings_defaults.chat_model
            ),
            "rewrite_model": (
                settings_defaults.local_rewrite_model
                if settings_defaults.use_local_rewrite_model
                else settings_defaults.chat_model
            ),
            "llm_judge_enabled": enable_llm_judge,
            "eval_execution_mode": eval_execution_mode,
            "requested_concurrency": requested_concurrency,
        },
        expanded=True,
    )
    if source_config["source_type"] == "dataset" and source_config["dataset_split"] != eval_split:
        st.warning(
            f"Knowledge base split is `{source_config['dataset_split']}`, "
            f"but evaluation split is `{eval_split}`. Rebuild the index with the same split for a fair test."
        )


def run_case(case_question: str, chat_history: list[dict] | None = None) -> dict:
    started_at = time.perf_counter()
    result = ask_rag(
        case_question,
        chat_history=chat_history,
        collection_name=source_config["collection_name"],
        source_type=source_config["source_type"],
        source_url=source_config["source_url"],
        uploaded_files_dir=source_config["uploaded_files_dir"],
        custom_source_name=source_config["custom_source_name"],
        dataset_name=source_config["dataset_name"],
        dataset_split=source_config["dataset_split"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_mode=retrieval_mode,
        retrieve_k=retrieve_k,
        use_rerank=use_rerank,
        rerank_k=rerank_k,
        chat_backend=chat_backend,
    )
    metrics = dict(result.get("metrics", {}))
    metrics["total_time"] = time.perf_counter() - started_at
    result["metrics"] = metrics
    return result


if run_single:
    parsed_history = parse_history_lines(history_input)
    result = run_case(question, chat_history=parsed_history)

    st.divider()
    st.subheader("Single Test Result")
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Rewrite", f"{result['metrics'].get('rewrite_time', 0.0):.2f}s")
    metric_right.metric("Retrieval", f"{result['metrics'].get('retrieval_time', 0.0):.2f}s")
    metric_left.metric("Rerank", f"{result['metrics'].get('rerank_time', 0.0):.2f}s")
    metric_right.metric("Total", f"{result['metrics'].get('total_time', 0.0):.2f}s")

    st.write("**Answer**")
    st.write(result["answer"])

    details_left, details_right = st.columns(2)
    with details_left:
        st.write("**Rewrite details**")
        st.json(
            {
                "rewrite_used": result.get("rewrite_used", False),
                "rewrite_decision": result.get("rewrite_decision", "skip_no_history"),
                "rewritten_question": result.get("rewritten_question", question),
            },
            expanded=True,
        )
    with details_right:
        st.write("**Retrieved evidence**")
        evidence_rows = []
        for index, doc in enumerate(result["retrieved_docs"], start=1):
            evidence_rows.append(
                {
                    "rank": index,
                    "score": (
                        result["rerank_scores"][index - 1]
                        if index - 1 < len(result["rerank_scores"])
                        else None
                    ),
                    "chunk_id": getattr(doc, "metadata", {}).get("chunk_id"),
                    "source": getattr(doc, "metadata", {}).get("filename")
                    or getattr(doc, "metadata", {}).get("dataset")
                    or getattr(doc, "metadata", {}).get("source"),
                    "preview": getattr(doc, "page_content", "")[:160],
                }
            )
        st.dataframe(evidence_rows, use_container_width=True, hide_index=True)

    with st.expander("Raw response payload", expanded=False):
        st.code(json.dumps(result["config"], ensure_ascii=False, indent=2), language="json")
        for index, doc in enumerate(result["retrieved_docs"], start=1):
            st.caption(f"Chunk {index}: {getattr(doc, 'metadata', {})}")
            st.write(getattr(doc, "page_content", ""))


if run_batch:
    cases = parse_batch_cases(batch_input)
    if not cases:
        st.warning("No valid batch cases found.")
    else:
        st.divider()
        st.subheader("Batch Test Result")
        batch_rows = []
        rolling_history: list[dict] = []

        for case in cases:
            chat_history = None
            if batch_history_mode == "Carry previous user question" and rolling_history:
                chat_history = rolling_history[-3:]
            result = run_case(case["question"], chat_history=chat_history)
            batch_rows.append(
                {
                    "case_id": case["case_id"],
                    "question": case["question"],
                    "expected": case["expected"],
                    "rewrite_used": result.get("rewrite_used", False),
                    "rewrite_decision": result.get("rewrite_decision", "skip_no_history"),
                    "rewritten_question": result.get("rewritten_question", case["question"]),
                    "top_chunk_id": (
                        getattr(result["retrieved_docs"][0], "metadata", {}).get("chunk_id")
                        if result["retrieved_docs"]
                        else None
                    ),
                    "top_source": (
                        getattr(result["retrieved_docs"][0], "metadata", {}).get("filename")
                        or getattr(result["retrieved_docs"][0], "metadata", {}).get("dataset")
                        or getattr(result["retrieved_docs"][0], "metadata", {}).get("source")
                        if result["retrieved_docs"]
                        else None
                    ),
                    "answer_preview": result["answer"][:120],
                    "rewrite_s": round(result["metrics"].get("rewrite_time", 0.0), 2),
                    "retrieval_s": round(result["metrics"].get("retrieval_time", 0.0), 2),
                    "rerank_s": round(result["metrics"].get("rerank_time", 0.0), 2),
                    "total_s": round(result["metrics"].get("total_time", 0.0), 2),
                }
            )
            if batch_history_mode == "Carry previous user question":
                rolling_history = [
                    {"role": "user", "content": case["question"]},
                    {"role": "assistant", "content": result["answer"]},
                ]

        st.dataframe(batch_rows, use_container_width=True, hide_index=True)


if run_eval:
    if source_config["source_type"] != "dataset":
        st.warning("Dataset evaluation is only available when Knowledge source is set to dataset.")
    else:
        st.divider()
        st.subheader("Dataset Evaluation Result")
        with st.spinner("Running sampled evaluation cases..."):
            samples = load_dataset_samples(
                source_config["dataset_name"],
                eval_split,
                eval_sample_size,
            )
            effective_concurrency = requested_concurrency
            if eval_execution_mode == "serial":
                effective_concurrency = 1
            elif chat_backend == "local_qwen":
                effective_concurrency = min(requested_concurrency, 2)

            eval_rows = []
            answer_hit_count = 0
            exact_match_count = 0.0
            f1_sum = 0.0
            top_chunk_hit_count = 0
            topk_hit_count = 0
            recall_at_k_sum = 0.0
            precision_at_k_sum = 0.0
            mrr_sum = 0.0
            relevance_sum = 0.0
            faithfulness_sum = 0.0
            rewrite_used_count = 0
            total_time_sum = 0.0
            wall_started_at = time.perf_counter()
            evaluated_items = []
            if effective_concurrency == 1:
                for sample in samples:
                    evaluated_items.append(evaluate_dataset_sample(sample))
            else:
                with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
                    future_map = {
                        executor.submit(evaluate_dataset_sample, sample): sample
                        for sample in samples
                    }
                    for future in as_completed(future_map):
                        evaluated_items.append(future.result())
            wall_time = time.perf_counter() - wall_started_at

            evaluated_items.sort(key=lambda item: item["sample"]["id"])

            for item in evaluated_items:
                sample = item["sample"]
                result = item["result"]
                retrieval_metrics = item["retrieval_metrics"]
                answer_hit_count += int(item["answer_hit"])
                exact_match_count += item["answer_em"]
                f1_sum += item["answer_f1"]
                top_chunk_hit_count += int(retrieval_metrics["top1_hit"])
                topk_hit_count += int(retrieval_metrics["topk_hit"])
                recall_at_k_sum += retrieval_metrics["recall_at_k"]
                precision_at_k_sum += retrieval_metrics["precision_at_k"]
                mrr_sum += retrieval_metrics["mrr"]
                relevance_sum += item["relevance_score"] or 0.0
                faithfulness_sum += item["faithfulness_score"] or 0.0
                rewrite_used_count += int(item["rewrite_used"])
                total_time_sum += result["metrics"].get("total_time", 0.0)

                eval_rows.append(
                    {
                        "id": sample["id"],
                        "question": sample["question"],
                        "gold_answers": " | ".join(sample["answers"][:3]),
                        "answer_hit": item["answer_hit"],
                        "answer_em": round(item["answer_em"], 2),
                        "answer_f1": round(item["answer_f1"], 2),
                        "top1_hit": bool(retrieval_metrics["top1_hit"]),
                        "topk_hit": bool(retrieval_metrics["topk_hit"]),
                        "precision_at_k": round(retrieval_metrics["precision_at_k"], 2),
                        "recall_at_k": round(retrieval_metrics["recall_at_k"], 2),
                        "mrr": round(retrieval_metrics["mrr"], 2),
                        "first_hit_rank": retrieval_metrics["first_hit_rank"],
                        "relevance": item["relevance_score"],
                        "faithfulness": item["faithfulness_score"],
                        "rewrite_used": item["rewrite_used"],
                        "rewritten_question": result.get(
                            "rewritten_question",
                            sample["question"],
                        ),
                        "top_chunk_id": (
                            getattr(result["retrieved_docs"][0], "metadata", {}).get("chunk_id")
                            if result["retrieved_docs"]
                            else None
                        ),
                        "answer_preview": result["answer"][:160],
                        "total_s": round(result["metrics"].get("total_time", 0.0), 2),
                    }
                )

        summary_left, summary_mid, summary_right = st.columns(3)
        total_cases = max(len(eval_rows), 1)
        summary_left.metric("Cases", len(eval_rows))
        summary_mid.metric("Answer Hit Rate", f"{answer_hit_count / total_cases:.1%}")
        summary_right.metric("Top Chunk Hit Rate", f"{top_chunk_hit_count / total_cases:.1%}")

        summary_left.metric("Rewrite Used Rate", f"{rewrite_used_count / total_cases:.1%}")
        summary_mid.metric("Avg Total Time", f"{total_time_sum / total_cases:.2f}s")
        summary_right.metric("Split", eval_split)

        speed_left, speed_mid, speed_right = st.columns(3)
        speed_left.metric("Wall Time", f"{wall_time:.2f}s")
        speed_mid.metric("Throughput", f"{len(eval_rows) / max(wall_time, 1e-6):.2f} samples/s")
        speed_right.metric("Concurrency", effective_concurrency)

        retrieval_left, retrieval_mid, retrieval_right = st.columns(3)
        retrieval_left.metric("Top-k Hit Rate", f"{topk_hit_count / total_cases:.1%}")
        retrieval_mid.metric("Recall@k", f"{recall_at_k_sum / total_cases:.1%}")
        retrieval_right.metric("Precision@k", f"{precision_at_k_sum / total_cases:.1%}")

        quality_left, quality_mid, quality_right = st.columns(3)
        quality_left.metric("MRR", f"{mrr_sum / total_cases:.3f}")
        quality_mid.metric("Answer EM", f"{exact_match_count / total_cases:.1%}")
        quality_right.metric("Answer F1", f"{f1_sum / total_cases:.3f}")

        if enable_llm_judge:
            judge_left, judge_mid = st.columns(2)
            judge_left.metric("Relevance", f"{relevance_sum / total_cases:.1%}")
            judge_mid.metric("Faithfulness", f"{faithfulness_sum / total_cases:.1%}")

        st.dataframe(eval_rows, use_container_width=True, hide_index=True)
