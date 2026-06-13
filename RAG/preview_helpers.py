from index_builder import get_chunks_cache_path, load_chunks_cache
from knowledge_base import list_uploaded_file_entries, load_uploaded_document_file


def load_preview_context(settings):
    cache_path = get_chunks_cache_path(settings)
    chunks = load_chunks_cache(settings)
    uploaded_files = []
    uploaded_docs_error = None

    if settings.uploaded_files_dir:
        try:
            uploaded_files = list_uploaded_file_entries(settings.uploaded_files_dir)
        except RuntimeError as exc:
            uploaded_docs_error = str(exc)

    return {
        "cache_path": cache_path,
        "chunks": chunks,
        "uploaded_files": uploaded_files,
        "uploaded_docs_error": uploaded_docs_error,
    }


def load_selected_file_preview(selected_entry):
    if not selected_entry:
        return [], None

    try:
        selected_docs = load_uploaded_document_file(
            selected_entry["source"],
            strict_pdf=False,
        )
        return selected_docs, None
    except RuntimeError as exc:
        return [], str(exc)
