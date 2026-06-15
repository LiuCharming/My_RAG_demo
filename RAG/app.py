import time

import streamlit as st

from rag_service import build_index, delete_knowledge_base, prepare_rag_response
from rag_settings import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_SOURCE_URL,
    UPLOADS_DIR,
    sanitize_name,
)
from ui_helpers import ensure_chat_state, list_custom_knowledge_bases, save_uploaded_files


st.set_page_config(page_title="RAG System", page_icon="R", layout="wide")

st.title("RAG System")
st.caption("Local vector database, retrieval, reranking, and answer generation.")


with st.sidebar:
    st.subheader("Index Settings")
    source_type = st.selectbox(
        "Knowledge source",
        options=["web", "dataset", "custom"],
        index=0,
        help="web = article demo, dataset = CMRC2018, custom = your own named knowledge base",
    )
    collection_name = "rag_demo"
    custom_source_name = None
    source_url = DEFAULT_SOURCE_URL
    dataset_name = DEFAULT_DATASET_NAME
    dataset_split = DEFAULT_DATASET_SPLIT
    uploaded_files = []
    if source_type == "web":
        source_url = st.text_input("Web article URL", value=DEFAULT_SOURCE_URL)
        uploaded_files = st.file_uploader(
            "Or upload local files",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            help="If files are uploaded, indexing will use these files first. Supported: txt, md, pdf.",
        )
        collection_name = "web_demo"
    elif source_type == "custom":
        existing_custom_bases = list_custom_knowledge_bases()
        custom_mode = st.radio(
            "Custom knowledge base",
            options=["Select existing", "Create new"],
            horizontal=False,
        )
        if custom_mode == "Select existing" and existing_custom_bases:
            custom_source_name = st.selectbox(
                "Existing knowledge bases",
                options=existing_custom_bases,
            )
        else:
            default_name = existing_custom_bases[0] if existing_custom_bases else "python_manual"
            custom_source_name = st.text_input(
                "Knowledge base name",
                value=default_name,
                help="Use a stable name such as python_manual or company_docs.",
            )
        if custom_mode == "Select existing" and not existing_custom_bases:
            st.info("No custom knowledge base yet. Create one by uploading files below.")
        uploaded_files = st.file_uploader(
            "Upload knowledge files",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            help="Upload files to create or replace the corpus of this custom knowledge base. Supported: txt, md, pdf.",
        )
        collection_name = f"custom_{sanitize_name(custom_source_name)}"
        if custom_mode == "Select existing" and custom_source_name:
            if st.button("Delete knowledge base", use_container_width=True):
                uploads_target = UPLOADS_DIR / sanitize_name(collection_name)
                delete_knowledge_base(
                    collection_name=collection_name,
                    source_type="custom",
                    uploaded_files_dir=str(uploads_target),
                    custom_source_name=custom_source_name,
                )
                st.success(f"Deleted knowledge base: {custom_source_name}")
                st.rerun()
    else:
        dataset_name = st.text_input(
            "Dataset name",
            value=DEFAULT_DATASET_NAME,
            help="For example: hfl/cmrc2018",
        )
        dataset_split = st.text_input(
            "Dataset split",
            value=DEFAULT_DATASET_SPLIT,
            help="For example: train, test, validation",
        )
        collection_name = f"dataset_{sanitize_name(dataset_name)}_{sanitize_name(dataset_split)}"
    chunk_size = st.slider(
        "Chunk size",
        min_value=300,
        max_value=2000,
        value=1000,
        step=100,
    )
    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
    )

    st.subheader("Retrieval Settings")
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        options=["vector", "bm25", "hybrid"],
        index=0,
    )
    retrieve_k = st.slider("Retrieve k", min_value=2, max_value=20, value=10, step=1)
    use_rerank = st.checkbox("Use rerank", value=True)
    rerank_k = st.slider("Rerank k", min_value=1, max_value=10, value=2, step=1)

    if st.button("Rebuild Local Index", use_container_width=True):
        uploads_target = UPLOADS_DIR / sanitize_name(collection_name)
        uploaded_files_dir = save_uploaded_files(uploaded_files, uploads_target)
        if uploaded_files_dir is None and uploads_target.exists():
            uploaded_files_dir = str(uploads_target)
        with st.spinner("Rebuilding local vector database..."):
            build_result = build_index(
                collection_name=collection_name,
                source_type=source_type,
                source_url=source_url,
                uploaded_files_dir=uploaded_files_dir,
                custom_source_name=custom_source_name,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                retrieval_mode=retrieval_mode,
                retrieve_k=retrieve_k,
                use_rerank=use_rerank,
                rerank_k=rerank_k,
            )
        st.success(f"Index ready. Chunks: {len(build_result['splits'])}")

ensure_chat_state(st.session_state)


left, right = st.columns([2, 1], gap="large")

with left:
    history_container = st.container()
    live_container = st.container()

    with history_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

with right:
    st.subheader("Retrieved Evidence")
    st.page_link(
        "pages/Knowledge_Base_Preview.py",
        label="Open knowledge base preview",
        icon=":material/search:",
    )
    if st.session_state.last_config:
        st.json(st.session_state.last_config, expanded=True)

    rewrite_used = bool(st.session_state.last_metrics.get("rewrite_used")) if st.session_state.last_metrics else False
    if rewrite_used and st.session_state.last_rewritten_question:
        st.subheader("Search Query")
        st.caption("Question used for retrieval after multi-turn rewriting")
        st.write(st.session_state.last_rewritten_question)

    if st.session_state.last_metrics:
        st.subheader("Performance")
        metrics = st.session_state.last_metrics
        metric_left, metric_right = st.columns(2)
        metric_left.metric(
            "Rewrite",
            f"{metrics.get('rewrite_time', 0.0):.2f}s",
        )
        metric_right.metric(
            "Retrieval",
            f"{metrics.get('retrieval_time', 0.0):.2f}s",
        )
        metric_left.metric(
            "Rerank",
            f"{metrics.get('rerank_time', 0.0):.2f}s",
        )
        metric_right.metric(
            "Generation",
            f"{metrics.get('generation_time', 0.0):.2f}s",
        )
        metric_left.metric(
            "Total",
            f"{metrics.get('total_time', 0.0):.2f}s",
        )

    if not st.session_state.last_retrieved_docs:
        st.info("Evidence chunks will appear here after you ask a question.")
    else:
        for index, doc in enumerate(st.session_state.last_retrieved_docs, start=1):
            score = None
            if index - 1 < len(st.session_state.last_scores):
                score = st.session_state.last_scores[index - 1]
            title = f"Chunk {index}"
            if score is not None:
                title = f"{title} | rerank score: {score:.4f}"
            with st.expander(title, expanded=(index == 1)):
                st.caption(str(getattr(doc, "metadata", {})))
                st.write(getattr(doc, "page_content", str(doc)))


prompt = st.chat_input("Ask a question about the indexed knowledge base")

if prompt:
    uploads_target = UPLOADS_DIR / sanitize_name(collection_name)
    uploaded_files_dir = str(uploads_target) if uploads_target.exists() else None
    st.session_state.messages.append({"role": "user", "content": prompt})
    request_started_at = time.perf_counter()

    with live_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                result = prepare_rag_response(
                    prompt,
                    chat_history=st.session_state.messages[-6:],
                    collection_name=collection_name,
                    source_type=source_type,
                    source_url=source_url,
                    uploaded_files_dir=uploaded_files_dir,
                    custom_source_name=custom_source_name,
                    dataset_name=dataset_name,
                    dataset_split=dataset_split,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    retrieval_mode=retrieval_mode,
                    retrieve_k=retrieve_k,
                    use_rerank=use_rerank,
                    rerank_k=rerank_k,
                )
            generation_started_at = time.perf_counter()

            message_placeholder = st.empty()
            full_response = ""

            for text in result["pipeline"].stream_answer(
                prompt,
                result["retrieved_docs"],
            ):
                full_response += text
                message_placeholder.markdown(full_response + "|")

            message_placeholder.markdown(full_response)
            generation_time = time.perf_counter() - generation_started_at

    base_metrics = dict(result.get("metrics", {}))
    base_metrics["rewrite_used"] = result.get("rewrite_used", False)
    base_metrics["generation_time"] = generation_time
    base_metrics["total_time"] = time.perf_counter() - request_started_at

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
    st.session_state.last_retrieved_docs = result["retrieved_docs"]
    st.session_state.last_scores = result["rerank_scores"]
    st.session_state.last_config = result["config"]
    st.session_state.last_metrics = base_metrics
    st.session_state.last_rewritten_question = result.get("rewritten_question", prompt)
    st.rerun()
