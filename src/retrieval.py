from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = "index/psle_faiss"

# Module-level cache to avoid reloading index on every query
_cached_vectorstore = None
_cached_embeddings = None


def get_embeddings():
    """Return a HuggingFaceEmbeddings instance (runs locally, no API key needed).
    Uses module-level cache to avoid reinitializing the model on every call.
    """
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _cached_embeddings


def build_index(documents):
    """Create a FAISS vectorstore from documents and save it to disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"[retrieval] FAISS index built and saved to '{INDEX_PATH}' ({len(documents)} documents).")
    return vectorstore


def load_index():
    """Load a previously saved FAISS index from disk.
    Uses module-level cache so the index is only loaded once per session.
    """
    global _cached_vectorstore
    if _cached_vectorstore is None:
        embeddings = get_embeddings()
        _cached_vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print(f"[retrieval] FAISS index loaded from '{INDEX_PATH}'.")
    return _cached_vectorstore


def get_retriever(k=4):
    """Load the FAISS index and return a retriever."""
    vectorstore = load_index()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_with_scores(question: str, k: int = 4):
    """
    Retrieve documents with their similarity scores.
    
    FAISS L2 distance: lower = more similar. We convert to a 0-1 similarity
    score using 1/(1+distance) so higher = more similar.
    
    Args:
        question: The question to search for
        k: Number of documents to retrieve
    
    Returns:
        List of (document, similarity_score) tuples, where similarity_score
        is in [0, 1] with higher being more similar
    """
    vectorstore = load_index()
    results = vectorstore.similarity_search_with_score(question, k=k)
    # Convert L2 distance to similarity score: 1/(1+distance)
    return [(doc, 1.0 / (1.0 + score)) for doc, score in results]


def retrieve_by_topic(question: str, topic: str, k: int = 4):
    """
    Retrieve documents filtered by PSLE topic, with similarity scores.
    
    Fetches a larger candidate set, filters by topic, then returns top-k.
    Falls back to unfiltered results if too few topic matches are found.
    
    Args:
        question: The question to search for
        topic: PSLE topic key (e.g., "percentage", "rate")
        k: Number of documents to retrieve
    
    Returns:
        List of (document, similarity_score) tuples filtered by topic
    """
    vectorstore = load_index()
    
    # Fetch larger candidate set for filtering
    candidate_size = k * 8
    all_results = vectorstore.similarity_search_with_score(question, k=candidate_size)
    
    # Filter by topic and convert scores
    topic_results = [
        (doc, 1.0 / (1.0 + score))
        for doc, score in all_results
        if doc.metadata.get("topic") == topic
    ]
    
    # If we found enough topic-specific results, use them
    if len(topic_results) >= k:
        return topic_results[:k]
    
    # If not enough topic matches, return what we have plus top general results
    if topic_results:
        return topic_results
    
    # Fallback: no topic matches at all, return general results
    print(f"[retrieval] Warning: No documents found for topic '{topic}', falling back to general search.")
    return [(doc, 1.0 / (1.0 + score)) for doc, score in all_results[:k]]
