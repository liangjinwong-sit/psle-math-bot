from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = "index/psle_faiss"
DEFAULT_K = 4
DEFAULT_SCORE_THRESHOLD = 1.0


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_index(documents):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"[retrieval] FAISS index built and saved to '{INDEX_PATH}' ({len(documents)} documents).")
    return vectorstore


def load_index():
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[retrieval] FAISS index loaded from '{INDEX_PATH}'.")
    return vectorstore


def retrieve_relevant_docs(question: str, k: int = DEFAULT_K, score_threshold: float = DEFAULT_SCORE_THRESHOLD):
    vectorstore = load_index()
    results = vectorstore.similarity_search_with_score(question, k=k)

    filtered = []
    for doc, score in results:
        if score <= score_threshold:
            doc.metadata["score"] = float(score)
            filtered.append(doc)

    print(
        f"[retrieval] Retrieved {len(results)} docs, kept {len(filtered)} docs "
        f"with threshold <= {score_threshold}."
    )
    return filtered


def get_retriever(k=DEFAULT_K):
    vectorstore = load_index()
    return vectorstore.as_retriever(search_kwargs={"k": k})
