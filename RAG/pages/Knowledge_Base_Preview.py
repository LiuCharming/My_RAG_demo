import streamlit as st

from preview_helpers import load_preview_context, load_selected_file_preview
from rag_service import (
    delete_file_from_custom_knowledge_base,
    delete_knowledge_base,
    rename_custom_knowledge_base,
)
from rag_settings import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_SOURCE_URL,
    sanitize_name,
)
from ui_helpers import get_uploaded_files_dir, list_custom_knowledge_bases, make_settings


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
            collection_name = f"custom_{sanitize_name(custom_source_name or 'python_manual')}"
            uploads_dir = get_uploaded_files_dir(collection_name) or ""
            renamed_value = st.text_input(
                "Rename knowledge base",
                value=custom_source_name,
                help="Enter a new name and click Rename to move this knowledge base.",
            )
            if st.button("Rename knowledge base", use_container_width=True):
                if renamed_value.strip():
                    rename_custom_knowledge_base(
                        old_name=custom_source_name,
                        new_name=renamed_value,
                    )
                    st.success(f"Renamed knowledge base to: {renamed_value}")
                    st.rerun()
            if st.button("Delete this knowledge base", use_container_width=True):
                delete_knowledge_base(
                    collection_name=collection_name,
                    source_type="custom",
                    uploaded_files_dir=uploads_dir,
                    custom_source_name=custom_source_name,
                )
                st.success(f"Deleted knowledge base: {custom_source_name}")
                st.rerun()
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

preview_context = load_preview_context(settings)
cache_path = preview_context["cache_path"]
chunks = preview_context["chunks"]
uploaded_files = preview_context["uploaded_files"]
uploaded_docs_error = preview_context["uploaded_docs_error"]

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
            "uploaded_document_count": len(uploaded_files),
        },
        expanded=True,
    )

    st.subheader("Files")
    if uploaded_docs_error:
        st.warning(uploaded_docs_error)
    if not uploaded_files:
        st.info("No uploaded files for this knowledge base.")
    else:
        file_names = [entry["filename"] for entry in uploaded_files]
        selected_file = st.selectbox("Uploaded files", options=file_names)
        selected_entry = next(
            (entry for entry in uploaded_files if entry["filename"] == selected_file),
            None,
        )
        selected_docs, file_preview_error = load_selected_file_preview(selected_entry)
        if file_preview_error:
            uploaded_docs_error = file_preview_error

        if uploaded_docs_error:
            st.warning(uploaded_docs_error)
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
            if (
                settings.source_type == "custom"
                and custom_source_name
                and st.button("Delete this file", use_container_width=True)
            ):
                result = delete_file_from_custom_knowledge_base(
                    custom_source_name=custom_source_name,
                    filename=selected_file,
                )
                if result == "deleted_knowledge_base":
                    st.success("Deleted the last file. The knowledge base was removed.")
                else:
                    st.success(f"Deleted file: {selected_file}")
                st.rerun()

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
