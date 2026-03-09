from src.ingest import get_all_documents
from src.retrieval import build_index

if __name__ == "__main__":
    # Build index from:
    # 1. Your curated markdown notes (PRIORITY)
    # 2. PSLE-Notes-LLM.pdf (if exists)
    # 3. GSM8K dataset (optional, set to False by default)
    
    documents = get_all_documents(
        include_markdown=True,    # Your curated topic notes
        include_pdf=True,          # PSLE-Notes-LLM.pdf
        include_gsm8k=False,       # Set to True only if you want supplementary examples
        pdf_paths=["PSLE-Notes-LLM.pdf"]  # Add more PDFs here if needed
    )
    
    print(f"\n📊 Building FAISS index with {len(documents)} documents...")
    build_index(documents)
    print("✅ Index build complete. Ready to run the app!")
