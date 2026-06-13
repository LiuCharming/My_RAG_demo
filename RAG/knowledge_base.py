import bs4
import requests
from datasets import load_dataset
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_settings import RAGSettings


def load_web_documents(url: str) -> list[Document]:
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(
        response.text,
        "html.parser",
        parse_only=bs4_strainer,
    )
    return [Document(page_content=soup.get_text(), metadata={"source": url})]


def load_cmrc2018_documents(dataset_name: str, split: str) -> list[Document]:
    dataset = load_dataset(dataset_name, split=split)
    seen_contexts = set()
    documents = []

    for row in dataset:
        context = row["context"]
        if context in seen_contexts:
            continue
        seen_contexts.add(context)
        documents.append(
            Document(
                page_content=context,
                metadata={
                    "dataset": dataset_name,
                    "split": split,
                },
            )
        )

    return documents


def load_uploaded_documents(folder: str) -> list[Document]:
    documents = []
    base = Path(folder)
    if not base.exists():
        return documents

    for path in sorted(base.glob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                },
            )
        )
    return documents


def load_source_documents(settings: RAGSettings) -> list[Document]:
    if settings.source_type == "dataset":
        return load_cmrc2018_documents(
            dataset_name=settings.dataset_name,
            split=settings.dataset_split,
        )
    if settings.source_type == "custom" and settings.uploaded_files_dir:
        return load_uploaded_documents(settings.uploaded_files_dir)
    if settings.uploaded_files_dir:
        uploaded_documents = load_uploaded_documents(settings.uploaded_files_dir)
        if uploaded_documents:
            return uploaded_documents
    return load_web_documents(settings.source_url)


def split_documents(documents: list[Document], settings: RAGSettings) -> list[Document]:
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
