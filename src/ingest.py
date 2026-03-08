import os
<<<<<<< Updated upstream
from datasets import load_dataset
=======
from glob import glob
>>>>>>> Stashed changes
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def load_pdf_docs(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = get_splitter()
    chunks = splitter.split_documents(pages)

    filename = os.path.basename(pdf_path)
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = filename
        chunk.metadata["type"] = "pdf"
        chunk.metadata["chunk_id"] = i

    print(f"[ingest] Loaded {len(chunks)} chunks from PDF '{filename}'.")
    return chunks


def load_markdown_docs(notes_dir: str):
    splitter = get_splitter()
    docs = []

    md_files = sorted(glob(os.path.join(notes_dir, "*.md")))
    if not md_files:
        print(f"[ingest] No markdown files found in '{notes_dir}'.")
        return docs

    for path in md_files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        filename = os.path.basename(path)
        raw_doc = Document(
            page_content=text,
            metadata={
                "source": filename,
                "type": "markdown",
                "path": path,
            },
        )

        chunks = splitter.split_documents([raw_doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = filename
            chunk.metadata["type"] = "markdown"
            chunk.metadata["chunk_id"] = i

        docs.extend(chunks)
        print(f"[ingest] Loaded {len(chunks)} chunks from markdown '{filename}'.")

    return docs


def get_all_documents(notes_dir: str = "data/notes", pdf_paths=None):
    if pdf_paths is None:
        pdf_paths = []

    all_docs = []

    if os.path.isdir(notes_dir):
        all_docs.extend(load_markdown_docs(notes_dir))
    else:
        print(f"[ingest] Notes directory '{notes_dir}' not found.")

    for path in pdf_paths:
        if os.path.exists(path):
            all_docs.extend(load_pdf_docs(path))
        else:
            print(f"[ingest] Skipping missing PDF '{path}'.")

    print(f"[ingest] Total documents collected: {len(all_docs)}")
    return all_docs
