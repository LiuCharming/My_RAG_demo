from pathlib import Path

from rag_settings import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_SOURCE_URL,
    RAGSettings,
    UPLOADS_DIR,
    sanitize_name,
)


def list_custom_knowledge_bases() -> list[str]:
    if not UPLOADS_DIR.exists():
        return []
    names = []
    for path in sorted(UPLOADS_DIR.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if name.startswith("custom_"):
            name = name[len("custom_") :]
        names.append(name)
    return names


def save_uploaded_files(uploaded_files, target_dir: Path) -> str | None:
    if not uploaded_files:
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    for existing in target_dir.glob("*"):
        if existing.is_file():
            existing.unlink()
    for uploaded_file in uploaded_files:
        target = target_dir / Path(uploaded_file.name).name
        target.write_bytes(uploaded_file.getbuffer())
    return str(target_dir)


def get_collection_name(
    source_type: str,
    custom_source_name: str | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
) -> str:
    if source_type == "custom":
        return f"custom_{sanitize_name(custom_source_name or 'python_manual')}"
    if source_type == "dataset":
        return f"dataset_{sanitize_name(dataset_name)}_{sanitize_name(dataset_split)}"
    return "web_demo"


def get_uploaded_files_dir(collection_name: str) -> str | None:
    uploads_dir = UPLOADS_DIR / sanitize_name(collection_name)
    return str(uploads_dir) if uploads_dir.exists() else None


def make_settings(
    source_type: str,
    source_url: str = DEFAULT_SOURCE_URL,
    custom_source_name: str | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
) -> RAGSettings:
    collection_name = get_collection_name(
        source_type=source_type,
        custom_source_name=custom_source_name,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    return RAGSettings(
        collection_name=collection_name,
        source_type=source_type,
        source_url=source_url,
        uploaded_files_dir=get_uploaded_files_dir(collection_name),
        custom_source_name=custom_source_name,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )


def ensure_chat_state(session_state) -> None:
    if "messages" not in session_state:
        session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I am your RAG assistant. "
                    "Build the local index first, then ask a question."
                ),
            }
        ]

    if "last_retrieved_docs" not in session_state:
        session_state.last_retrieved_docs = []

    if "last_scores" not in session_state:
        session_state.last_scores = []

    if "last_config" not in session_state:
        session_state.last_config = {}

    if "last_metrics" not in session_state:
        session_state.last_metrics = {}

    if "last_rewritten_question" not in session_state:
        session_state.last_rewritten_question = ""

    if "rewrite_context_messages" not in session_state:
        session_state.rewrite_context_messages = []

    if "rewrite_topic_anchor" not in session_state:
        session_state.rewrite_topic_anchor = ""
