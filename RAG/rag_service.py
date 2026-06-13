from functools import lru_cache
from typing import TYPE_CHECKING

from rag_settings import RAGSettings

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
