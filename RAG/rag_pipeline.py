import os
import time
from functools import lru_cache
from pathlib import Path

from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from index_builder import load_chunks_cache, load_vector_store
from rag_settings import MODEL_CACHE_DIR, RAGSettings
from langsmith import traceable

try:
    import jieba
except ImportError:  # pragma: no cover - fallback for environments without jieba
    jieba = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local env loading
    load_dotenv = None


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
        Document(
            page_content=page_content,
            metadata=metadata or {},
        )
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
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY is not set.")


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
        self.llm = get_llm(settings.chat_model)

    @traceable(name="rewrite_question")
    def rewrite_question(self, question: str, chat_history: list[dict] | None = None) -> str:
        if not chat_history:
            return question

        normalized_history = [
            message
            for message in chat_history
            if isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and str(message.get("content", "")).strip()
        ]
        if not normalized_history:
            return question

        recent_history = normalized_history[-6:]
        history_lines = []
        for message in recent_history:
            role = "用户" if message["role"] == "user" else "助手"
            history_lines.append(f"{role}: {message['content']}")

        prompt = (
            "请根据对话历史，将当前问题改写成一个完整、明确、适合知识检索的独立问题。"
            "如果当前问题已经足够完整，就保持原意。"
            "不要回答问题，只输出改写后的问题。\n\n"
            f"对话历史：\n{'\n'.join(history_lines)}\n\n"
            f"当前问题：{question}\n\n"
            "改写后的问题："
        )

        rewritten_question = self.llm.invoke(prompt).content.strip()
        return rewritten_question or question

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
        prompt = (
            "Answer only with the provided context. "
            "If the context is insufficient, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        return self.llm.invoke(prompt).content
    #@traceable(run_type="chain")
    def ask(self, question: str, chat_history: list[dict] | None = None) -> dict:
        prepared = self.prepare_answer(question, chat_history=chat_history)
        full_answer = "".join(
            self.stream_answer(question, prepared["retrieved_docs"])
        )
        return {
            "answer": full_answer,
            "retrieved_docs": prepared["retrieved_docs"],
            "candidate_docs": prepared["candidate_docs"],
            "rerank_scores": prepared["rerank_scores"],
            "rewritten_question": prepared.get("rewritten_question", question),
            "metrics": prepared.get("metrics", {}),
        }
    @traceable(run_type="chain")
    def prepare_answer(self, question: str, chat_history: list[dict] | None = None) -> dict:
        prepare_started_at = time.perf_counter()

        rewrite_started_at = time.perf_counter()
        rewritten_question = self.rewrite_question(question, chat_history=chat_history)
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
        prompt = (
            "Answer only with the provided context. "
            "If the context is insufficient, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )

        for chunk in self.llm.stream(prompt):
            text = getattr(chunk, "content", None)
            if text:
                yield text
