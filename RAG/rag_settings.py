import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db" / "chroma_demo"
UPLOADS_DIR = PROJECT_ROOT / "RAG" / "uploads"
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
MODEL_CACHE_DIR = Path("F:/huggingface_models")
DEFAULT_SOURCE_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
DEFAULT_DATASET_NAME = "hfl/cmrc2018"
DEFAULT_DATASET_SPLIT = "train"


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return cleaned.strip("_").lower() or "rag_demo"


@dataclass
class RAGSettings:
    collection_name: str = "rag_demo"
    source_type: str = "web"
    source_url: str = DEFAULT_SOURCE_URL
    uploaded_files_dir: str | None = None
    custom_source_name: str | None = None
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_split: str = DEFAULT_DATASET_SPLIT
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_mode: str = "vector"
    retrieve_k: int = 10
    use_rerank: bool = True
    rerank_k: int = 2
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    chat_model: str = "deepseek-v4-flash"
