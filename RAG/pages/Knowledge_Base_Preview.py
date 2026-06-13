import streamlit as st

from index_builder import get_chunks_cache_path, load_chunks_cache
from knowledge_base import load_uploaded_documents
from rag_settings import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_SOURCE_URL,
    RAGSettings,
    UPLOADS_DIR,
    sanitize_name,
)


def list_custom_knowledge_bases() -> list[str]:
    if not UPLOADS_DIR.exists():
        return []
    names = []
    for path in sorted(UPLOADS_DIR.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if name.startswith("custom_"):
            name = name[len("custom_") :]
        names.append(name)
    return names


def make_settings(
    source_type: str,
    source_url: str,
    custom_source_name: str | None,
    dataset_name: str,
    dataset_split: str,
) -> RAGSettings:
    if source_type == "custom":
        collection_name = f"custom_{sanitize_name(custom_source_name or 'python_manual')}"
        uploaded_files_dir = str(UPLOADS_DIR / sanitize_name(collection_name))
        return RAGSettings(
            collection_name=collection_name,
            source_type="custom",
            source_url=source_url,
            uploaded_files_dir=uploaded_files_dir,
            custom_source_name=custom_source_name,
        )
    if source_type == "dataset":
        return RAGSettings(
            collection_name=f"dataset_{sanitize_name(dataset_name)}_{sanitize_name(dataset_split)}",
            source_type="dataset",
            source_url=source_url,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )
    uploaded_dir = UPLOADS_DIR / sanitize_name("web_demo")
    uploaded_files_dir = str(uploaded_dir) if uploaded_dir.exists() else None
    return RAGSettings(
        collection_name="web_demo",
        source_type="web",
        source_url=source_url,
        uploaded_files_dir=uploaded_files_dir,
    )


st.set_page_config(page_title="Knowledge Base Preview", page_icon="K", layout="wide")

st.title("Knowledge Base Preview")
st.caption("Preview uploaded files, cached chunks, and collection details.")
st.page_link("app.py", label="Back to RAG chat", icon=":material/chat:")

with st.sidebar:
    source_type = st.selectbox(
        "Knowledge source",
        options=["web", "dataset", "custom"],
        index=2,
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
    elif source_type == "custom":
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

settings = make_settings(
    source_type=source_type,
    source_url=source_url,
    custom_source_name=custom_source_name,
    dataset_name=dataset_name,
    dataset_split=dataset_split,
)

cache_path = get_chunks_cache_path(settings)
chunks = load_chunks_cache(settings)
uploaded_docs = []
uploaded_docs_error = None
if settings.uploaded_files_dir:
    try:
        uploaded_docs = load_uploaded_documents(
            settings.uploaded_files_dir,
            strict_pdf=False,
        )
    except RuntimeError as exc:
        uploaded_docs_error = str(exc)

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Overview")
    st.json(
        {
            "collection_name": settings.collection_name,
            "source_type": settings.source_type,
            "source_url": settings.source_url,
            "dataset_name": settings.dataset_name,
            "dataset_split": settings.dataset_split,
            "uploaded_files_dir": settings.uploaded_files_dir,
            "chunks_cache_path": str(cache_path),
            "chunk_count": len(chunks),
            "uploaded_document_count": len(uploaded_docs),
        },
        expanded=True,
    )

    st.subheader("Files")
    if uploaded_docs_error:
        st.warning(uploaded_docs_error)
    if not uploaded_docs:
        st.info("No uploaded files for this knowledge base.")
    else:
        file_names = sorted(
            {
                doc.metadata.get("filename", "unknown")
                for doc in uploaded_docs
            }
        )
        selected_file = st.selectbox("Uploaded files", options=file_names)
        selected_docs = [
            doc for doc in uploaded_docs if doc.metadata.get("filename") == selected_file
        ]
        if selected_docs:
            selected_doc = selected_docs[0]
            st.caption(selected_doc.metadata.get("source", ""))
            file_type = selected_doc.metadata.get("file_type")
            page_count = selected_doc.metadata.get("page_count")
            extraction_method = selected_doc.metadata.get("extraction_method")
            page_start = selected_doc.metadata.get("page_start")
            page_end = selected_doc.metadata.get("page_end")
            if file_type or page_count:
                details = []
                if file_type:
                    details.append(f"type: {file_type}")
                if page_count:
                    details.append(f"pages: {page_count}")
                if page_start and page_end:
                    details.append(f"range: {page_start}-{page_end}")
                if extraction_method:
                    details.append(f"method: {extraction_method}")
                st.caption(" | ".join(details))

            preview_text = "\n\n".join(
                doc.page_content for doc in selected_docs[: min(len(selected_docs), 3)]
            )
            if len(selected_docs) > 3:
                preview_text += "\n\n[Preview limited to the first 3 pages]"
            st.text_area(
                "File preview",
                value=preview_text[:4000],
                height=320,
            )

with right:
    st.subheader("Chunk Preview")
    if not chunks:
        st.warning("No chunk cache found yet. Rebuild the index from the main page first.")
    else:
        search_term = st.text_input(
            "Search in chunks",
            placeholder="Enter a keyword such as rudder, 舵, python",
        ).strip()

        if search_term:
            lowered = search_term.lower()
            preview_chunks = [
                chunk
                for chunk in chunks
                if lowered in chunk.page_content.lower()
                or lowered in str(chunk.metadata).lower()
            ]
            st.caption(f"Matched chunks: {len(preview_chunks)} / {len(chunks)}")
        else:
            preview_chunks = chunks
            st.caption(f"Total chunks: {len(chunks)}")

        available_sources = sorted(
            {
                chunk.metadata.get("filename")
                or chunk.metadata.get("dataset")
                or chunk.metadata.get("source")
                or "unknown"
                for chunk in preview_chunks
            }
        )
        selected_source = st.selectbox(
            "Filter by source",
            options=["All files"] + available_sources,
            index=0,
        )
        if selected_source != "All files":
            preview_chunks = [
                chunk
                for chunk in preview_chunks
                if (
                    chunk.metadata.get("filename")
                    or chunk.metadata.get("dataset")
                    or chunk.metadata.get("source")
                    or "unknown"
                )
                == selected_source
            ]
            st.caption(f"Filtered chunks: {len(preview_chunks)}")

        st.subheader("Chunk List")
        if not preview_chunks:
            st.info("No matching chunks found for the current search.")
        else:
            max_preview = min(len(preview_chunks), 20)
            if max_preview == 1:
                selected_chunk = preview_chunks[0]
            else:
                selected_index = st.slider(
                    "Chunk index",
                    min_value=0,
                    max_value=max_preview - 1,
                    value=0,
                    step=1,
                )
                selected_chunk = preview_chunks[selected_index]

            st.caption(str(selected_chunk.metadata))
            st.text_area(
                "Chunk content",
                value=selected_chunk.page_content,
                height=420,
            )

        chunk_rows = []
        for index, chunk in enumerate(preview_chunks[:50]):
            chunk_rows.append(
                {
                    "index": index,
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "page_start": chunk.metadata.get("page_start")
                    or chunk.metadata.get("page_number"),
                    "page_end": chunk.metadata.get("page_end")
                    or chunk.metadata.get("page_number"),
                    "method": chunk.metadata.get("extraction_method"),
                    "source": chunk.metadata.get("filename")
                    or chunk.metadata.get("dataset")
                    or chunk.metadata.get("source"),
                    "preview": chunk.page_content[:120].replace("\n", " "),
                }
            )
        if chunk_rows:
            st.dataframe(chunk_rows, use_container_width=True, hide_index=True)
