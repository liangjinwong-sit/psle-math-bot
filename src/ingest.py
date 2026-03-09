import os
import glob
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def load_markdown_notes(notes_dir: str = "data/notes"):
    """Load curated PSLE markdown notes with topic-aware metadata."""
    docs = []
    markdown_files = glob.glob(os.path.join(notes_dir, "*.md"))
    
    # Markdown splitter that respects headers
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    for md_path in markdown_files:
        filename = os.path.basename(md_path)
        
        # Extract topic from filename (e.g., "topic_percentage.md" -> "percentage")
        topic = filename.replace("topic_", "").replace(".md", "")
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by headers
        splits = md_splitter.split_text(content)
        
        for i, doc in enumerate(splits):
            doc.metadata["source"] = filename
            doc.metadata["topic"] = topic
            doc.metadata["chunk_id"] = i
            docs.append(doc)
    
    print(f"[ingest] Loaded {len(docs)} chunks from {len(markdown_files)} markdown files.")
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
        # Try to infer topic from content
        chunk.metadata["topic"] = "general"
    print(f"[ingest] Loaded {len(chunks)} chunks from '{filename}'.")
    return chunks


def load_gsm8k_docs(limit: int = 500):
    """Load GSM8K dataset (optional, for supplementary examples)."""
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    docs = []
    
    # Limit GSM8K to avoid overwhelming PSLE content
    for i, row in enumerate(dataset):
        if i >= limit:
            break
        solution = row["answer"].split("####")[0].strip()
        page_content = f"Question: {row['question']}\nSolution: {solution}"
        doc = Document(
            page_content=page_content,
            metadata={"source": "gsm8k", "id": i, "topic": "arithmetic"},
        )
        docs.append(doc)
    print(f"[ingest] Loaded {len(docs)} documents from GSM8K dataset (limited).")
    return docs


def get_all_documents(
    include_markdown: bool = True,
    include_pdf: bool = True,
    include_gsm8k: bool = False,
    pdf_paths: list = None,
):
    """
    Combine all data sources into one list.
    
    Priority:
    1. Curated markdown notes (highest relevance)
    2. PSLE PDF notes
    3. GSM8K (optional, for supplementary examples)
    """
    all_docs = []
    
    # PRIORITY 1: Your curated PSLE notes (markdown)
    if include_markdown:
        all_docs.extend(load_markdown_notes())
    
    # PRIORITY 2: PSLE PDF
    if include_pdf and pdf_paths:
        for path in pdf_paths:
            if os.path.exists(path):
                all_docs.extend(load_pdf_docs(path))
    
    # PRIORITY 3: GSM8K (optional, limited to avoid diluting PSLE content)
    if include_gsm8k:
        all_docs.extend(load_gsm8k_docs(limit=500))
    
    print(f"[ingest] Total documents collected: {len(all_docs)}")
    return all_docs
