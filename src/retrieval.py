from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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


def get_retriever(k=4, topic_filter=None):
    """
    Load the FAISS index and return a retriever.
    
    Args:
        k: Number of documents to retrieve
        topic_filter: Optional topic to filter by (e.g., "percentage", "fractions_decimals")
    
    Returns:
        Retriever with optional topic filtering
    """
    vectorstore = load_index()
    
    if topic_filter:
        # Create retriever with metadata filter
        return vectorstore.as_retriever(
            search_kwargs={
                "k": k * 3,  # Retrieve more, then filter
                "filter": {"topic": topic_filter}
            }
        )
    else:
        return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_by_topic(question: str, topic: str, k: int = 4):
    """
    Retrieve documents specifically from a topic.
    
    Args:
        question: The question to search for
        topic: PSLE topic key (e.g., "percentage", "rate")
        k: Number of documents to retrieve
    
    Returns:
        List of documents matching the query and topic
    """
    vectorstore = load_index()
    
    # Similarity search with score
    all_docs = vectorstore.similarity_search_with_score(question, k=k*5)
    
    # Filter by topic
    topic_docs = [(doc, score) for doc, score in all_docs if doc.metadata.get("topic") == topic]
    
    # Return top k from filtered results
    return [doc for doc, score in topic_docs[:k]]
