import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_gsm8k_docs():
    """Load the GSM8K dataset and convert each row into a LangChain Document."""
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    docs = []
    for i, row in enumerate(dataset):
        # Strip everything after "####" in the answer for cleaner text
        solution = row["answer"].split("####")[0].strip()
        page_content = f"Question: {row['question']}\nSolution: {solution}"
        doc = Document(
            page_content=page_content,
            metadata={"source": "gsm8k", "id": i, "topic": "arithmetic"},
        )
        docs.append(doc)
    print(f"[ingest] Loaded {len(docs)} documents from GSM8K dataset.")
    return docs


def load_pdf_docs(pdf_path: str):
    """Load a PDF and split it into chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    filename = os.path.basename(pdf_path)
    for chunk in chunks:
        chunk.metadata["source"] = filename
    print(f"[ingest] Loaded {len(chunks)} chunks from '{filename}'.")
    return chunks


def get_all_documents(pdf_paths: list = []):
    """Combine GSM8K docs and all PDF docs into one list."""
    all_docs = load_gsm8k_docs()
    for path in pdf_paths:
        all_docs.extend(load_pdf_docs(path))
    print(f"[ingest] Total documents collected: {len(all_docs)}")
    return all_docs
