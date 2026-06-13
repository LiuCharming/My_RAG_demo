from functools import lru_cache
from typing import TYPE_CHECKING

from pathlib import Path

from rag_settings import RAGSettings, UPLOADS_DIR, sanitize_name

if TYPE_CHECKING:
    from rag_pipeline import RAGPipeline


@lru_cache(maxsize=8)
def get_pipeline(
    collection_name: str,
    source_type: str,
    source_url: str,
    uploaded_files_dir: str | None,
    custom_source_name: str | None,
    dataset_name: str,
    dataset_split: str,
    chunk_size: int,
    chunk_overlap: int,
    retrieval_mode: str,
    retrieve_k: int,
    use_rerank: bool,
    rerank_k: int,
) -> "RAGPipeline":
    from rag_pipeline import RAGPipeline

    settings = RAGSettings(
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
    return RAGPipeline(settings)


def build_index(
    collection_name: str = "rag_demo",
    source_type: str = "web",
    source_url: str = RAGSettings.source_url,
    uploaded_files_dir: str | None = None,
    custom_source_name: str | None = None,
    dataset_name: str = RAGSettings.dataset_name,
    dataset_split: str = RAGSettings.dataset_split,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    retrieval_mode: str = "vector",
    retrieve_k: int = 10,
    use_rerank: bool = True,
    rerank_k: int = 2,
) -> dict:
    from index_builder import rebuild_vector_store

    settings = RAGSettings(
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
    get_pipeline.cache_clear()
    return rebuild_vector_store(settings)


def delete_knowledge_base(
    collection_name: str,
    source_type: str = "custom",
    uploaded_files_dir: str | None = None,
    custom_source_name: str | None = None,
    dataset_name: str = RAGSettings.dataset_name,
    dataset_split: str = RAGSettings.dataset_split,
) -> None:
    from index_builder import delete_knowledge_base_storage

    settings = RAGSettings(
        collection_name=collection_name,
        source_type=source_type,
        uploaded_files_dir=uploaded_files_dir,
        custom_source_name=custom_source_name,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    get_pipeline.cache_clear()
    delete_knowledge_base_storage(settings)


def rename_custom_knowledge_base(
    old_name: str,
    new_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    retrieval_mode: str = "vector",
    retrieve_k: int = 10,
    use_rerank: bool = True,
    rerank_k: int = 2,
) -> str:
    from index_builder import delete_knowledge_base_storage

    old_collection_name = f"custom_{sanitize_name(old_name)}"
    new_collection_name = f"custom_{sanitize_name(new_name)}"
    old_upload_dir = UPLOADS_DIR / sanitize_name(old_collection_name)
    new_upload_dir = UPLOADS_DIR / sanitize_name(new_collection_name)

    if not old_upload_dir.exists():
        raise FileNotFoundError(f"Knowledge base not found: {old_name}")
    if new_upload_dir.exists() and old_upload_dir != new_upload_dir:
        raise FileExistsError(f"Knowledge base already exists: {new_name}")

    if old_upload_dir != new_upload_dir:
        old_upload_dir.rename(new_upload_dir)

    old_settings = RAGSettings(
        collection_name=old_collection_name,
        source_type="custom",
        uploaded_files_dir=str(new_upload_dir),
        custom_source_name=old_name,
    )
    delete_knowledge_base_storage(old_settings, delete_uploads=False)

    build_index(
        collection_name=new_collection_name,
        source_type="custom",
        uploaded_files_dir=str(new_upload_dir),
        custom_source_name=new_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_mode=retrieval_mode,
        retrieve_k=retrieve_k,
        use_rerank=use_rerank,
        rerank_k=rerank_k,
    )
    get_pipeline.cache_clear()
    return new_collection_name


def delete_file_from_custom_knowledge_base(
    custom_source_name: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    retrieval_mode: str = "vector",
    retrieve_k: int = 10,
    use_rerank: bool = True,
    rerank_k: int = 2,
) -> str:
    collection_name = f"custom_{sanitize_name(custom_source_name)}"
    upload_dir = UPLOADS_DIR / sanitize_name(collection_name)
    file_path = upload_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    file_path.unlink()

    remaining_files = [path for path in upload_dir.glob("*") if path.is_file()]
    if not remaining_files:
        delete_knowledge_base(
            collection_name=collection_name,
            source_type="custom",
            uploaded_files_dir=str(upload_dir),
            custom_source_name=custom_source_name,
        )
        return "deleted_knowledge_base"

    build_index(
        collection_name=collection_name,
        source_type="custom",
        uploaded_files_dir=str(upload_dir),
        custom_source_name=custom_source_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_mode=retrieval_mode,
        retrieve_k=retrieve_k,
        use_rerank=use_rerank,
        rerank_k=rerank_k,
    )
    get_pipeline.cache_clear()
    return "reindexed"


def ask_rag(
    question: str,
    collection_name: str = "rag_demo",
    source_type: str = "web",
    source_url: str = RAGSettings.source_url,
    uploaded_files_dir: str | None = None,
    custom_source_name: str | None = None,
    dataset_name: str = RAGSettings.dataset_name,
    dataset_split: str = RAGSettings.dataset_split,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    retrieval_mode: str = "vector",
    retrieve_k: int = 10,
    use_rerank: bool = True,
    rerank_k: int = 2,
) -> dict:
    pipeline = get_pipeline(
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
    result = pipeline.ask(question)
    return {
        "answer": result["answer"],
        "retrieved_docs": result["retrieved_docs"],
        "rerank_scores": result["rerank_scores"],
        "metrics": result.get("metrics", {}),
        "config": {
            "collection_name": collection_name,
            "source_url": source_url,
            "uploaded_files_dir": uploaded_files_dir,
            "custom_source_name": custom_source_name,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_mode": retrieval_mode,
            "retrieve_k": retrieve_k,
            "use_rerank": use_rerank,
            "rerank_k": rerank_k,
        },
    }


def prepare_rag_response(
    question: str,
    collection_name: str = "rag_demo",
    source_type: str = "web",
    source_url: str = RAGSettings.source_url,
    uploaded_files_dir: str | None = None,
    custom_source_name: str | None = None,
    dataset_name: str = RAGSettings.dataset_name,
    dataset_split: str = RAGSettings.dataset_split,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    retrieval_mode: str = "vector",
    retrieve_k: int = 10,
    use_rerank: bool = True,
    rerank_k: int = 2,
) -> dict:
    pipeline = get_pipeline(
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
    prepared = pipeline.prepare_answer(question)
    return {
        "pipeline": pipeline,
        "retrieved_docs": prepared["retrieved_docs"],
        "rerank_scores": prepared["rerank_scores"],
        "metrics": prepared.get("metrics", {}),
        "config": {
            "collection_name": collection_name,
            "source_url": source_url,
            "uploaded_files_dir": uploaded_files_dir,
            "custom_source_name": custom_source_name,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_mode": retrieval_mode,
            "retrieve_k": retrieve_k,
            "use_rerank": use_rerank,
            "rerank_k": rerank_k,
        },
    }
