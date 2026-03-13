import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

INDEX_PATH = "index/psle_faiss"

# Number of candidates fetched before reranking/diversity selection.
RETRIEVAL_CANDIDATE_MULTIPLIER = 5
MAX_RETRIEVAL_CANDIDATES = 60

# When filtering by topic, we fetch this many times the requested k
# as candidates before filtering, to increase the chance of finding
# enough on-topic results.
TOPIC_FILTER_CANDIDATE_MULTIPLIER = 8

# Lightweight lexical rerank weight combined with embedding similarity.
LEXICAL_RERANK_WEIGHT = 0.20

# Keep at least one full chunk in final context when available.
MIN_FULL_CHUNKS = 1

# Remove instruction-like lines from retrieved context to reduce
# prompt-injection risk from untrusted source text.
INJECTION_LINE_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"follow\s+these\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"assistant\s*:",
    r"user\s*:",
]

# Module-level cache to avoid reloading index on every query
_cached_vectorstore = None
_cached_embeddings = None


def _normalize_query(question: str) -> str:
    """Normalize user query before retrieval."""
    if question is None:
        return ""
    return re.sub(r"\s+", " ", str(question)).strip()


def _tokenize(text: str) -> set:
    """Simple tokenizer for lexical overlap reranking."""
    return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def _distance_to_similarity(distance: float) -> float:
    """Convert FAISS L2 distance (lower is better) to similarity (higher is better)."""
    return 1.0 / (1.0 + float(distance))


def _strip_injection_lines(text: str) -> str:
    """Remove suspicious instruction-like lines from untrusted retrieved text."""
    safe_lines = []
    for raw_line in (text or "").split("\n"):
        line = raw_line.strip()
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in INJECTION_LINE_PATTERNS):
            continue
        safe_lines.append(raw_line)
    return "\n".join(safe_lines).strip()


def _sanitize_doc_content(doc: Document) -> Document:
    """Return a Document copy with sanitized page_content."""
    cleaned_content = _strip_injection_lines(doc.page_content)
    if cleaned_content == doc.page_content:
        return doc

    updated_metadata = dict(doc.metadata)
    updated_metadata["sanitized"] = True
    return Document(page_content=cleaned_content, metadata=updated_metadata)


def _rerank_results(question: str, raw_results):
    """Rerank retrieval candidates using semantic + lexical signal."""
    query_tokens = _tokenize(question)
    reranked = []
    for doc, distance in raw_results:
        semantic_score = _distance_to_similarity(distance)
        doc_tokens = _tokenize(doc.page_content)
        overlap = len(query_tokens.intersection(doc_tokens)) / max(1, len(query_tokens))
        combined_score = min(1.0, semantic_score + LEXICAL_RERANK_WEIGHT * overlap)
        reranked.append((_sanitize_doc_content(doc), combined_score))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked


def _select_diverse_results(ranked_results, k: int):
    """Select a diverse top-k set (prefer full chunk presence + parent diversity)."""
    if not ranked_results:
        return []

    selected = []
    selected_chunk_ids = set()
    selected_parents = set()

    # Pass 1: ensure at least MIN_FULL_CHUNKS full documents when available.
    full_candidates = [
        (doc, score) for doc, score in ranked_results
        if doc.metadata.get("chunk_type", "full") == "full"
    ]
    for doc, score in full_candidates[:MIN_FULL_CHUNKS]:
        chunk_id = doc.metadata.get("chunk_id")
        parent_id = doc.metadata.get("parent_id", chunk_id)
        selected.append((doc, score))
        selected_chunk_ids.add(chunk_id)
        selected_parents.add(parent_id)
        if len(selected) >= k:
            return selected

    # Pass 2: add top candidates with parent diversity first.
    for doc, score in ranked_results:
        chunk_id = doc.metadata.get("chunk_id")
        parent_id = doc.metadata.get("parent_id", chunk_id)
        if chunk_id in selected_chunk_ids:
            continue
        if parent_id in selected_parents:
            continue
        selected.append((doc, score))
        selected_chunk_ids.add(chunk_id)
        selected_parents.add(parent_id)
        if len(selected) >= k:
            return selected

    # Pass 3: fill remaining slots by score regardless of parent overlap.
    for doc, score in ranked_results:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id in selected_chunk_ids:
            continue
        selected.append((doc, score))
        selected_chunk_ids.add(chunk_id)
        if len(selected) >= k:
            break

    return selected


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
    clean_question = _normalize_query(question)

    # Stage 1: fetch a larger candidate pool.
    candidate_size = min(MAX_RETRIEVAL_CANDIDATES, max(k * RETRIEVAL_CANDIDATE_MULTIPLIER, k))
    raw_results = vectorstore.similarity_search_with_score(clean_question, k=candidate_size)

    # Stage 2: rerank and select a diverse final top-k.
    ranked = _rerank_results(clean_question, raw_results)
    return _select_diverse_results(ranked, k)


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
    clean_question = _normalize_query(question)

    # Fetch larger candidate set for topic filtering + reranking.
    candidate_size = min(MAX_RETRIEVAL_CANDIDATES, max(k * TOPIC_FILTER_CANDIDATE_MULTIPLIER, k))
    all_results = vectorstore.similarity_search_with_score(clean_question, k=candidate_size)

    # Prefer topic-specific candidates first.
    topic_raw_results = [
        (doc, score)
        for doc, score in all_results
        if doc.metadata.get("topic") == topic
    ]

    if topic_raw_results:
        ranked_topic = _rerank_results(clean_question, topic_raw_results)
        selected_topic = _select_diverse_results(ranked_topic, k)
        if selected_topic:
            return selected_topic

    # Fallback: no topic matches found, use best general results.
    print(f"[retrieval] Warning: No documents found for topic '{topic}', falling back to general search.")
    ranked_general = _rerank_results(clean_question, all_results)
    return _select_diverse_results(ranked_general, k)
