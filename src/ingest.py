from datasets import load_dataset
from langchain_core.documents import Document
from src.topic_classifier import classify_question


def load_gsm8k_docs(split="train", limit=None):
    """
    Load the GSM8K dataset and classify into PSLE topics.
    
    This dataset contains 8.5K high quality grade school math word problems
    with step-by-step solutions. Each question is automatically classified
    into one of 6 PSLE math topics using keyword-based classification.
    
    Args:
        split: "train" or "test"
        limit: Maximum number of documents to load (None = load all)
    
    Returns:
        List of LangChain Documents with question, solution, and topic metadata
    
    Raises:
        ConnectionError: If dataset cannot be downloaded
    """
    try:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
    except Exception as e:
        raise ConnectionError(
            f"Failed to load GSM8K dataset ({split} split). "
            f"Check your internet connection.\nError: {e}"
        ) from e
    
    docs = []
    topic_counts = {}
    
    max_docs = len(dataset) if limit is None else min(limit, len(dataset))
    
    for i in range(max_docs):
        row = dataset[i]
        
        # Classify question into PSLE topic
        topic = classify_question(row["question"])
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Extract the solution (remove the final answer after ####)
        full_answer = row["answer"]
        solution_steps = full_answer.split("####")[0].strip()
        final_answer = full_answer.split("####")[1].strip() if "####" in full_answer else ""
        
        # Create document with question and solution
        page_content = (
            f"Question: {row['question']}\n\n"
            f"Solution:\n{solution_steps}\n\n"
            f"Final Answer: {final_answer}"
        )
        
        doc = Document(
            page_content=page_content,
            metadata={
                "source": "gsm8k",
                "split": split,
                "id": i,
                "topic": topic,
                "question": row["question"],
                "answer": final_answer,
            },
        )
        docs.append(doc)
    
    print(f"[ingest] Loaded {len(docs)} documents from GSM8K {split} split.")
    print(f"[ingest] Topic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {topic}: {count} ({count / len(docs) * 100:.1f}%)")
    
    return docs


def get_all_documents(use_train=True, use_test=False, train_limit=None, test_limit=None):
    """
    Load documents from GSM8K dataset.
    
    Args:
        use_train: Include training set (7473 examples)
        use_test: Include test set (1319 examples)
        train_limit: Limit training examples (None = all)
        test_limit: Limit test examples (None = all)
    
    Returns:
        Combined list of documents from requested splits
    """
    all_docs = []
    
    if use_train:
        all_docs.extend(load_gsm8k_docs(split="train", limit=train_limit))
    
    if use_test:
        all_docs.extend(load_gsm8k_docs(split="test", limit=test_limit))
    
    print(f"[ingest] Total documents collected: {len(all_docs)}")
    return all_docs
