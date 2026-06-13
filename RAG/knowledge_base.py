import bs4
import requests
from datasets import load_dataset
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ocr_support import extract_text_from_image_bytes
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


def looks_garbled_text(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True

    if len(cleaned) < 40:
        return False

    allowed_ranges = [
        (0x4E00, 0x9FFF),   # CJK
        (0x3400, 0x4DBF),   # CJK Extension A
        (0x0041, 0x005A),   # A-Z
        (0x0061, 0x007A),   # a-z
        (0x0030, 0x0039),   # 0-9
    ]
    allowed_chars = set(
        " \n\t.,;:!?()[]{}<>/\\|@#$%^&*-_=+'\"`~，。；：？！、（）《》【】"
    )

    def is_allowed(char: str) -> bool:
        code = ord(char)
        if char in allowed_chars:
            return True
        return any(start <= code <= end for start, end in allowed_ranges)

    allowed_count = sum(1 for char in cleaned if is_allowed(char))
    allowed_ratio = allowed_count / max(len(cleaned), 1)

    odd_characters = sum(
        1
        for char in cleaned
        if not is_allowed(char) and not char.isspace()
    )
    odd_ratio = odd_characters / max(len(cleaned), 1)

    return allowed_ratio < 0.72 or odd_ratio > 0.18

def read_pdf_documents(path: Path) -> list[Document]:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF upload requires the PyMuPDF package. Install requirements.txt first."
        ) from exc

    pdf = pymupdf.open(str(path))
    page_count = len(pdf)
    documents = []
    for page_index, page in enumerate(pdf, start=1):
        text = page.get_text("text", sort=True).strip()
        extraction_method = "text"

        if text and looks_garbled_text(text):
            try:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                ocr_text = extract_text_from_image_bytes(pixmap.tobytes("png")).strip()
                if ocr_text:
                    text = ocr_text
                    extraction_method = "ocr_fallback"
            except RuntimeError:
                extraction_method = "text_garbled_ocr_unavailable"
            except Exception:
                extraction_method = "text_garbled_ocr_failed"

        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "file_type": "pdf",
                    "page_number": page_index,
                    "page_count": page_count,
                    "extraction_method": extraction_method,
                },
            )
        )
    pdf.close()
    return documents


def read_uploaded_text(path: Path) -> tuple[str, dict]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return (
            path.read_text(encoding="utf-8", errors="ignore").strip(),
            {"file_type": suffix.lstrip(".")},
        )

    return "", {}


def load_uploaded_documents(folder: str, strict_pdf: bool = True) -> list[Document]:
    documents = []
    base = Path(folder)
    if not base.exists():
        return documents

    for path in sorted(base.glob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        if path.suffix.lower() == ".pdf":
            try:
                pdf_documents = read_pdf_documents(path)
            except RuntimeError:
                if strict_pdf:
                    raise
                pdf_documents = [
                    Document(
                        page_content="PDF preview is unavailable because PyMuPDF is not installed.",
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "file_type": "pdf",
                            "preview_error": "missing_pymupdf",
                        },
                    )
                ]
            documents.extend(pdf_documents)
            continue

        text, extra_metadata = read_uploaded_text(path)
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    **extra_metadata,
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
