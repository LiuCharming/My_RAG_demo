import os
import time
from functools import lru_cache
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langsmith import traceable
from sentence_transformers import CrossEncoder

from index_builder import load_chunks_cache, load_vector_store
from rag_settings import HF_CACHE_DIR, MODEL_CACHE_DIR, RAGSettings

try:
    import jieba
except ImportError:  # pragma: no cover - fallback for environments without jieba
    jieba = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local env loading
    load_dotenv = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional local rewrite model
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


def tokenize_for_bm25(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if jieba is not None:
        return [token for token in jieba.lcut(text) if token.strip()]
    return [char for char in text if not char.isspace()]


def load_documents_from_vector_store(vector_store) -> list[Document]:
    payload = vector_store.get(include=["documents", "metadatas"])
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    return [
        Document(page_content=page_content, metadata=metadata or {})
        for page_content, metadata in zip(documents, metadatas)
        if page_content
    ]


def load_local_env() -> None:
    if load_dotenv is None:
        return
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=False)
    load_dotenv(project_root / ".env.local", override=False)


def ensure_runtime_env() -> None:
    load_local_env()
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "ls-quickstart")
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY is not set.")


def normalize_chat_history(chat_history: list[dict] | None) -> list[dict]:
    return [
        message
        for message in chat_history or []
        if isinstance(message, dict)
        and message.get("role") in {"user", "assistant"}
        and str(message.get("content", "")).strip()
    ]


@lru_cache(maxsize=8)
def get_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name, cache_folder=str(MODEL_CACHE_DIR))


@lru_cache(maxsize=8)
def get_llm(model_name: str) -> ChatDeepSeek:
    ensure_runtime_env()
    return ChatDeepSeek(
        model=model_name,
        temperature=0,
        timeout=300,
        max_tokens=5000,
    )


@lru_cache(maxsize=2)
def get_local_rewrite_model(model_name: str):
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        raise RuntimeError("transformers or torch is not installed.")

    ensure_runtime_env()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(MODEL_CACHE_DIR),
        local_files_only=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(MODEL_CACHE_DIR),
        torch_dtype="auto",
        device_map="auto",
        local_files_only=False,
    )
    return tokenizer, model


def generate_with_local_model(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
) -> str:
    tokenizer, model = get_local_rewrite_model(model_name)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def run_local_rewrite(
    model_name: str,
    history_text: str,
    question: str,
) -> str:
    rewritten = generate_with_local_model(
        model_name,
        system_prompt=(
            "你是一个查询改写器。"
            "请把当前问题改写成适合检索的独立问题。"
            "如果当前问题本身已经完整、明确、不依赖上下文，就原样输出当前问题。"
            "不要回答问题，只输出最终检索问题。"
        ),
        user_prompt=f"对话历史：\n{history_text}\n\n当前问题：{question}",
        max_new_tokens=64,
    )
    return rewritten or question


class RAGPipeline:
    def __init__(self, settings: RAGSettings):
        self.settings = settings
        self.vector_store = load_vector_store(settings)
        self.chunk_documents = load_chunks_cache(settings)
        if not self.chunk_documents:
            self.chunk_documents = load_documents_from_vector_store(self.vector_store)
        self.bm25_retriever = None
        if self.chunk_documents:
            self.bm25_retriever = BM25Retriever.from_documents(
                self.chunk_documents,
                preprocess_func=tokenize_for_bm25,
            )
            self.bm25_retriever.k = self.settings.retrieve_k
        self.reranker = (
            get_reranker(settings.reranker_model) if settings.use_rerank else None
        )
        self.llm = None if settings.chat_backend == "local_qwen" else get_llm(settings.chat_model)

    @traceable(name="rewrite_question")
    def rewrite_question(
        self,
        question: str,
        chat_history: list[dict] | None = None,
    ) -> tuple[str, bool, str]:
        normalized_history = normalize_chat_history(chat_history)
        if len(normalized_history) < 2:
            return question, False, "skip_no_history"

        recent_history = normalized_history[-6:]
        history_lines = []
        for message in recent_history:
            role = "用户" if message["role"] == "user" else "助手"
            history_lines.append(f"{role}: {message['content']}")

        history_text = "\n".join(history_lines)

        rewritten_question = question
        if self.settings.use_local_rewrite_model:
            try:
                rewritten_question = run_local_rewrite(
                    self.settings.local_rewrite_model,
                    history_text,
                    question,
                )
            except Exception:
                rewritten_question = question
        else:
            prompt = (
                "请根据对话历史，将当前问题改写成一个完整、明确、适合知识检索的独立问题。"
                "如果当前问题本身已经完整、明确、不依赖上下文，就原样输出当前问题。"
                "不要回答问题，只输出最终检索问题。\n\n"
                f"对话历史：\n{history_text}\n\n"
                f"当前问题：{question}\n\n"
                "最终检索问题："
            )
            rewritten_question = self.llm.invoke(prompt).content.strip() or question

        rewrite_used = rewritten_question != question
        rewrite_decision = "rewrite" if rewrite_used else "skip"
        return rewritten_question, rewrite_used, rewrite_decision

    @traceable(run_type="retriever")
    def retrieve_vector(self, question: str):
        return self.vector_store.similarity_search(
            question,
            k=self.settings.retrieve_k,
        )

    @traceable(run_type="retriever")
    def retrieve_bm25(self, question: str):
        if self.bm25_retriever is None:
            return []
        return self.bm25_retriever.invoke(question)

    @traceable(run_type="retriever")
    def retrieve(self, question: str):
        if self.settings.retrieval_mode == "bm25":
            return self.retrieve_bm25(question)

        if self.settings.retrieval_mode == "hybrid":
            vector_docs = self.retrieve_vector(question)
            bm25_docs = self.retrieve_bm25(question)
            merged_docs = []
            seen_keys = set()
            for doc in vector_docs + bm25_docs:
                key = (
                    doc.page_content,
                    tuple(sorted((doc.metadata or {}).items())),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_docs.append(doc)
            return merged_docs[: self.settings.retrieve_k]

        return self.retrieve_vector(question)

    @traceable(name="rerank")
    def rerank(self, question: str, candidate_docs: list):
        if not candidate_docs:
            return []
        pairs = [[question, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(
            zip(candidate_docs, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return scored_docs[: self.settings.rerank_k]

    def answer(self, question: str, retrieved_docs: list) -> str:
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        system_prompt = "Answer only with the provided context. If the context is insufficient, say you do not know."
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        if self.settings.chat_backend == "local_qwen":
            return generate_with_local_model(
                self.settings.local_chat_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=512,
            )
        return self.llm.invoke(f"{system_prompt}\n\n{user_prompt}").content

    def ask(self, question: str, chat_history: list[dict] | None = None) -> dict:
        prepared = self.prepare_answer(question, chat_history=chat_history)
        full_answer = "".join(
            self.stream_answer(
                prepared.get("rewritten_question", question),
                prepared["retrieved_docs"],
            )
        )
        return {
            "answer": full_answer,
            "retrieved_docs": prepared["retrieved_docs"],
            "candidate_docs": prepared["candidate_docs"],
            "rerank_scores": prepared["rerank_scores"],
            "rewritten_question": prepared.get("rewritten_question", question),
            "rewrite_used": prepared.get("rewrite_used", False),
            "rewrite_decision": prepared.get("rewrite_decision", "skip_no_history"),
            "metrics": prepared.get("metrics", {}),
        }

    @traceable(run_type="chain")
    def prepare_answer(
        self,
        question: str,
        chat_history: list[dict] | None = None,
    ) -> dict:
        prepare_started_at = time.perf_counter()

        rewrite_started_at = time.perf_counter()
        rewritten_question, rewrite_used, rewrite_decision = self.rewrite_question(
            question,
            chat_history=chat_history,
        )
        rewrite_time = time.perf_counter() - rewrite_started_at

        retrieval_started_at = time.perf_counter()
        candidate_docs = self.retrieve(rewritten_question)
        retrieval_time = time.perf_counter() - retrieval_started_at

        if self.settings.use_rerank and self.reranker is not None:
            rerank_started_at = time.perf_counter()
            reranked = self.rerank(rewritten_question, candidate_docs)
            rerank_time = time.perf_counter() - rerank_started_at
            retrieved_docs = [doc for doc, _ in reranked]
            rerank_scores = [float(score) for _, score in reranked]
        else:
            retrieved_docs = candidate_docs[: self.settings.rerank_k]
            rerank_scores = []
            rerank_time = 0.0

        prepare_time = time.perf_counter() - prepare_started_at
        return {
            "retrieved_docs": retrieved_docs,
            "candidate_docs": candidate_docs,
            "rerank_scores": rerank_scores,
            "rewritten_question": rewritten_question,
            "rewrite_used": rewrite_used,
            "rewrite_decision": rewrite_decision,
            "metrics": {
                "rewrite_time": rewrite_time,
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "prepare_time": prepare_time,
            },
        }

    @traceable(metadata={"llm": "deepseek-v4-flash"})
    def stream_answer(self, question: str, retrieved_docs: list):
        if not retrieved_docs:
            yield "I do not know based on the current knowledge base."
            return

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        system_prompt = "Answer only with the provided context. If the context is insufficient, say you do not know."
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        if self.settings.chat_backend == "local_qwen":
            full_text = generate_with_local_model(
                self.settings.local_chat_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=512,
            )
            if not full_text:
                return
            chunk_size = 24
            for index in range(0, len(full_text), chunk_size):
                yield full_text[index : index + chunk_size]
            return

        prompt = f"{system_prompt}\n\n{user_prompt}"
        for chunk in self.llm.stream(prompt):
            text = getattr(chunk, "content", None)
            if text:
                yield text
