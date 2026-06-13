import os
import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from knowledge_base import load_source_documents, split_documents
from rag_settings import (
    HF_CACHE_DIR,
    MODEL_CACHE_DIR,
    RAGSettings,
    VECTOR_DB_DIR,
)


def ensure_env() -> None:
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))


def create_embeddings(settings: RAGSettings) -> HuggingFaceEmbeddings:
    ensure_env()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=str(MODEL_CACHE_DIR),
    )


def get_persist_directory() -> Path:
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    return VECTOR_DB_DIR


def get_chunks_cache_path(settings: RAGSettings) -> Path:
    cache_dir = VECTOR_DB_DIR.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"chunks_cache_{settings.collection_name}.json"


def load_vector_store(settings: RAGSettings) -> Chroma:
    embeddings = create_embeddings(settings)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(get_persist_directory()),
    )


def save_chunks_cache(splits: list[Document], settings: RAGSettings) -> None:
    cache_path = get_chunks_cache_path(settings)
    serialized = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata or {},
        }
        for doc in splits
    ]
    cache_path.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_chunks_cache(settings: RAGSettings) -> list[Document]:
    cache_path = get_chunks_cache_path(settings)
    if not cache_path.exists():
        return []
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=item["page_content"],
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]


def rebuild_vector_store(settings: RAGSettings) -> dict:
    documents = load_source_documents(settings)
    splits = split_documents(documents, settings)
    vector_store = load_vector_store(settings)
    vector_store.reset_collection()
    vector_store.add_documents(splits)
    save_chunks_cache(splits, settings)
    return {
        "documents": documents,
        "splits": splits,
        "vector_store": vector_store,
        "persist_directory": str(get_persist_directory()),
    }


def delete_knowledge_base_storage(
    settings: RAGSettings,
    delete_uploads: bool = True,
) -> None:
    try:
        vector_store = load_vector_store(settings)
        vector_store.delete_collection()
    except Exception:
        pass

    cache_path = get_chunks_cache_path(settings)
    if cache_path.exists():
        cache_path.unlink()

    if delete_uploads and settings.uploaded_files_dir:
        uploads_dir = Path(settings.uploaded_files_dir)
        if uploads_dir.exists():
            shutil.rmtree(uploads_dir, ignore_errors=True)
