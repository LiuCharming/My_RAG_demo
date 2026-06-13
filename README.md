# Local RAG System

A local RAG app built with Streamlit, Chroma, BM25 / vector / hybrid retrieval, optional reranking, and a small knowledge-base management workflow.

This project is aimed at two things:

1. a RAG chat app that we can actually use and test
2. a local knowledge-base tool that lets us inspect, preview, and debug indexed content

## What it supports

### Chat app

Main entry:

- [F:\ls-quickstart\RAG\app.py](F:\ls-quickstart\RAG\app.py)

Features:

- knowledge-base question answering
- streaming answers
- evidence chunk display
- rerank score display
- switchable knowledge sources

### Retrieval modes

The app supports three retrieval modes:

- `vector`
- `bm25`
- `hybrid`

In practice:

- `vector` is better for semantic matching
- `bm25` is better for direct keyword matching
- `hybrid` combines both

### Rerank toggle

The UI supports turning rerank on and off so we can compare:

- direct retrieval-to-answer
- retrieval followed by reranking

### Knowledge sources

The app currently supports:

- `web`: index a web article or uploaded local files
- `dataset`: index a Hugging Face dataset with configurable dataset name and split
- `custom`: create and reuse named local knowledge bases

### Custom knowledge bases

Custom knowledge bases can be named and reused, for example:

- `python_manual`
- `company_docs`
- `ml_notes`

Supported upload formats:

- `txt`
- `md`
- `pdf`

Uploaded files are stored locally and isolated by knowledge-base name.

### PDF upload

PDF upload is supported for both:

- `web` mode uploads
- `custom` knowledge-base uploads

PDFs are parsed page by page. That means:

- each page can become its own source document
- page-level metadata is preserved
- preview can show file type and page count

This makes PDF retrieval easier to inspect and debug than treating the whole file as one giant block.

### Knowledge-base preview page

Preview page:

- [F:\ls-quickstart\RAG\pages\Knowledge_Base_Preview.py](F:\ls-quickstart\RAG\pages\Knowledge_Base_Preview.py)

It supports:

- viewing knowledge-base overview information
- previewing uploaded files
- searching cached chunks
- filtering chunk results by source file
- browsing chunk metadata and content

This page is mainly for debugging and understanding what actually got indexed.

### OCR-RAG

OCR-related files:

- [F:\ls-quickstart\RAG\OCR_RAG.py](F:\ls-quickstart\RAG\OCR_RAG.py)
- [F:\ls-quickstart\RAG\ocr_support.py](F:\ls-quickstart\RAG\ocr_support.py)

This path is intended for image-to-text retrieval experiments and can be extended later.

## Project structure

Main working files:

```text
README.md
requirements.txt
.gitignore
RAG/
  app.py
  rag_settings.py
  knowledge_base.py
  index_builder.py
  rag_pipeline.py
  rag_service.py
  ocr_support.py
  OCR_RAG.py
  pages/
    Knowledge_Base_Preview.py
```

Responsibilities:

- `app.py`: main chat UI
- `rag_settings.py`: shared settings and paths
- `knowledge_base.py`: load and prepare source documents
- `index_builder.py`: build local vector store and chunk cache
- `rag_pipeline.py`: retrieval, BM25, hybrid logic, rerank, answer generation
- `rag_service.py`: service layer used by the UI
- `Knowledge_Base_Preview.py`: inspect uploaded files and chunk cache

## Installation

Use the virtual environment in this project if available.

Install dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PDF upload requires:

- `pypdf`

This is already included in `requirements.txt`.

## Environment variables

Do not hardcode API keys in the code.

Set environment variables before running:

- `DEEPSEEK_API_KEY`
- `LANGSMITH_API_KEY` if tracing is needed

Example:

```powershell
$env:DEEPSEEK_API_KEY="your_key"
$env:LANGSMITH_API_KEY="your_key"
```

You can also use a local `.env` file, but it should not be committed.

## Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\RAG\app.py --server.fileWatcherType none --server.headless true
```

Then open:

- [http://localhost:8501](http://localhost:8501)

## Typical workflow

### Build or rebuild an index

1. choose a knowledge source
2. configure retrieval mode and rerank settings
3. upload files if needed
4. click `Rebuild Local Index`

### Ask questions

1. build the index
2. ask a question in the chat
3. inspect evidence chunks on the right

### Preview indexed content

1. open the knowledge-base preview page
2. choose the same knowledge source
3. inspect files and chunks
4. search or filter chunk results

## Recent changes

Recent work included:

- vector / BM25 / hybrid retrieval support
- Chinese tokenization for BM25
- rerank toggle
- configurable dataset name and split
- custom knowledge-base creation and reuse
- knowledge-base preview page
- chunk search and source filtering
- PDF upload support
- page-level PDF metadata
- removal of hardcoded API keys from the main RAG path

## Security notes

Do not commit:

- `.env`
- API keys
- model cache
- vector database cache

Current `.gitignore` already excludes:

- `.venv/`
- `.hf_cache/`
- `vector_db/`
- `__pycache__/`
- `.env`

If a key was ever committed before, treat it as leaked:

1. rotate it
2. stop using the old key
3. clean local git history before pushing if needed

## Ideas for next steps

- `docx` upload support
- delete custom knowledge bases
- performance panel with retrieval / rerank / generation timing
- chunk highlighting in preview
- source-aware hybrid result labels
- Docker packaging

## Notes

This project is currently best described as:

- a local RAG prototype
- a retrieval experiment surface
- a knowledge-base inspection tool

It is already suitable for:

- local demos
- course projects
- retrieval experiments
- iterative knowledge-base tooling
