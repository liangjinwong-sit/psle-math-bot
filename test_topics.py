"""Quick test to verify topic metadata in FAISS index."""
from src.retrieval import load_index, retrieve_by_topic

# Test if index has topic metadata
vectorstore = load_index()
print("[OK] Index loaded successfully!")

# Test retrieval by topic
test_question = "What is 25% of 80?"
topic = "percentage"

print(f"\nTest Question: {test_question}")
print(f"Topic Filter: {topic}")

results = vectorstore.similarity_search(test_question, k=3)

print(f"\nRetrieved {len(results)} documents:")
for i, doc in enumerate(results, 1):
    doc_topic = doc.metadata.get("topic", "N/A")
    question_preview = doc.metadata.get("question", "N/A")[:60]
    print(f"\n{i}. Topic: {doc_topic}")
    print(f"   Question: {question_preview}...")

# Test topic filtering
print(f"\n\n--- Testing Topic Filtering ---")
topic_filtered = retrieve_by_topic(test_question, topic, k=3)
print(f"\n[OK] Topic-filtered retrieval returned {len(topic_filtered)} documents")
for i, (doc, score) in enumerate(topic_filtered, 1):
    doc_topic = doc.metadata.get("topic", "N/A")
    question_preview = doc.metadata.get("question", "N/A")[:60]
    print(f"\n{i}. Topic: {doc_topic}")
    print(f"   Question: {question_preview}...")

print("\n\n[OK] Topic metadata working correctly!" if all(doc.metadata.get("topic") == topic for doc, score in topic_filtered) else "\n\n[FAIL] Warning: Some documents don't have the expected topic")
