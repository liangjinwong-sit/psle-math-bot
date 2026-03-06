from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = "index/psle_faiss"


def get_embeddings():
    """Return a HuggingFaceEmbeddings instance (runs locally, no API key needed)."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_index(documents):
    """Create a FAISS vectorstore from documents and save it to disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"[retrieval] FAISS index built and saved to '{INDEX_PATH}' ({len(documents)} documents).")
    return vectorstore


def load_index():
    """Load a previously saved FAISS index from disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print(f"[retrieval] FAISS index loaded from '{INDEX_PATH}'.")
    return vectorstore


def get_retriever(k=4):
    """Load the FAISS index and return a retriever."""
    vectorstore = load_index()
    return vectorstore.as_retriever(search_kwargs={"k": k})
