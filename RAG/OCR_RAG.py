import os
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from ocr_support import extract_text_from_image_bytes

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local env loading
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
MODEL_CACHE_DIR = PROJECT_ROOT / "huggingface_models"
OCR_VECTOR_DB_DIR = PROJECT_ROOT / "vector_db" / "ocr_chroma"


@dataclass
class OCRRAGSettings:
    collection_name: str = "ocr_rag_demo"
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieve_k: int = 10
    rerank_k: int = 3
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    chat_model: str = "deepseek-v4-flash"


def load_local_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env.local", override=False)


def ensure_env() -> None:
    load_local_env()
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "ls-quickstart")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY is not set.")
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))


def create_embeddings(settings: OCRRAGSettings) -> HuggingFaceEmbeddings:
    ensure_env()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=str(MODEL_CACHE_DIR),
    )


def create_vector_store(settings: OCRRAGSettings) -> Chroma:
    OCR_VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = create_embeddings(settings)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(OCR_VECTOR_DB_DIR),
    )


def create_reranker(settings: OCRRAGSettings) -> CrossEncoder:
    return CrossEncoder(
        settings.reranker_model,
        cache_folder=str(MODEL_CACHE_DIR),
    )


def create_llm(settings: OCRRAGSettings) -> ChatDeepSeek:
    ensure_env()
    return ChatDeepSeek(
        model=settings.chat_model,
        temperature=0,
        timeout=300,
        max_tokens=4000,
    )


def load_ocr_documents(image_dir: str | Path) -> list[Document]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    documents = []
    supported_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in supported_suffixes:
            continue
        text = extract_text_from_image_bytes(image_path.read_bytes())
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_image": str(image_path),
                    "file_name": image_path.name,
                },
            )
        )
    return documents


def split_documents(documents: list[Document], settings: OCRRAGSettings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    splits = splitter.split_documents(documents)
    for index, doc in enumerate(splits):
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["chunk_id"] = index
    return splits


def build_ocr_index(
    image_dir: str | Path,
    settings: OCRRAGSettings | None = None,
) -> dict:
    settings = settings or OCRRAGSettings()
    documents = load_ocr_documents(image_dir)
    splits = split_documents(documents, settings)
    vector_store = create_vector_store(settings)
    vector_store.reset_collection()
    vector_store.add_documents(splits)
    return {
        "documents": documents,
        "splits": splits,
        "vector_store": vector_store,
        "persist_directory": str(OCR_VECTOR_DB_DIR),
    }


class OCRRAG:
    def __init__(self, settings: OCRRAGSettings | None = None):
        self.settings = settings or OCRRAGSettings()
        self.vector_store = create_vector_store(self.settings)
        self.reranker = create_reranker(self.settings)
        self.llm = create_llm(self.settings)

    def retrieve(self, question: str) -> list:
        return self.vector_store.similarity_search(
            question,
            k=self.settings.retrieve_k,
        )

    def rerank(self, question: str, candidate_docs: list) -> list[tuple]:
        pairs = [[question, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(pairs)
        return sorted(
            zip(candidate_docs, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[: self.settings.rerank_k]

    def answer(self, question: str, retrieved_docs: list) -> str:
        if not retrieved_docs:
            return "I do not know based on the current OCR knowledge base."

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompt = (
            "You are an OCR-RAG assistant. Answer only with the provided context. "
            "If the context is insufficient, answer with 'I do not know'.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        return self.llm.invoke(prompt).content

    def ask(self, question: str) -> dict:
        candidate_docs = self.retrieve(question)
        reranked = self.rerank(question, candidate_docs)
        retrieved_docs = [doc for doc, _ in reranked]
        rerank_scores = [float(score) for _, score in reranked]
        return {
            "answer": self.answer(question, retrieved_docs),
            "retrieved_docs": retrieved_docs,
            "rerank_scores": rerank_scores,
        }


if __name__ == "__main__":
    demo_dir = PROJECT_ROOT / "RAG" / "image"
    print("Building OCR index from:", demo_dir)
    try:
        build_result = build_ocr_index(demo_dir)
        print("OCR index ready.")
        print("Persist directory:", build_result["persist_directory"])
        print("Chunks:", len(build_result["splits"]))
    except RuntimeError as error:
        print(error)
