from src.ingest import get_all_documents
from src.retrieval import build_index

if __name__ == "__main__":
    documents = get_all_documents(notes_dir="data/notes")
    build_index(documents)
    print("Index build complete.")
