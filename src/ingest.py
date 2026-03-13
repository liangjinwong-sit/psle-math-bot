import re
from datasets import load_dataset
from langchain_core.documents import Document
from src.topic_classifier import classify_question


# Chunking modes for ingestion/indexing.
CHUNK_MODE_FULL = "full"
CHUNK_MODE_STEP = "step"
CHUNK_MODE_HYBRID = "hybrid"
VALID_CHUNK_MODES = {CHUNK_MODE_FULL, CHUNK_MODE_STEP, CHUNK_MODE_HYBRID}

# Step-window chunking defaults.
DEFAULT_STEP_WINDOW_SIZE = 3
DEFAULT_STEP_OVERLAP = 1


def _clean_text(text: str) -> str:
    """Normalize noisy text before storing and embedding."""
    if text is None:
        return ""
    # Remove GSM-style inline calculator traces: <<40/2=20>>
    cleaned = re.sub(r"<<[^>]*>>", "", str(text))
    # Trim trailing spaces but keep line boundaries for solution steps.
    lines = [line.rstrip() for line in cleaned.split("\n")]
    return "\n".join(lines).strip()


def _extract_solution_parts(full_answer: str):
    """Extract solution steps and final answer from GSM8K answer text."""
    answer_text = _clean_text(full_answer)
    if "####" in answer_text:
        solution_text, final_answer = answer_text.split("####", 1)
        final_answer = final_answer.strip()
    else:
        solution_text = answer_text
        final_answer = ""

    step_lines = [line.strip() for line in solution_text.split("\n") if line.strip()]
    return step_lines, "\n".join(step_lines), final_answer


def _make_full_doc(
    question: str,
    solution_text: str,
    final_answer: str,
    split: str,
    row_id: int,
    topic: str,
) -> Document:
    """Create one full-problem document (current legacy behavior)."""
    page_content = (
        f"Question: {question}\n\n"
        f"Solution:\n{solution_text}\n\n"
        f"Final Answer: {final_answer}"
    )
    return Document(
        page_content=page_content,
        metadata={
            "source": "gsm8k",
            "split": split,
            "id": row_id,
            "topic": topic,
            "question": question,
            "answer": final_answer,
            "chunk_type": "full",
            "chunk_id": f"{split}-{row_id}-full",
            "parent_id": f"{split}-{row_id}",
        },
    )


def _make_step_docs(
    question: str,
    step_lines: list,
    final_answer: str,
    split: str,
    row_id: int,
    topic: str,
    step_window_size: int,
    step_overlap: int,
):
    """Create solution-step window documents for finer-grained retrieval."""
    step_docs = []
    if not step_lines:
        return step_docs

    stride = max(1, step_window_size - step_overlap)
    for start_idx in range(0, len(step_lines), stride):
        end_idx = min(start_idx + step_window_size, len(step_lines))
        window_steps = step_lines[start_idx:end_idx]
        if not window_steps:
            continue

        numbered_steps = [
            f"Step {start_idx + offset + 1}: {text}"
            for offset, text in enumerate(window_steps)
        ]
        window_solution = "\n".join(numbered_steps)
        page_content = (
            f"Question: {question}\n\n"
            f"Solution:\n{window_solution}\n\n"
            f"Final Answer: {final_answer}"
        )

        step_docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "gsm8k",
                    "split": split,
                    "id": row_id,
                    "topic": topic,
                    "question": question,
                    "answer": final_answer,
                    "chunk_type": "solution_step",
                    "chunk_id": f"{split}-{row_id}-step-{start_idx + 1}-{end_idx}",
                    "parent_id": f"{split}-{row_id}",
                    "step_start": start_idx + 1,
                    "step_end": end_idx,
                    "num_steps": len(step_lines),
                },
            )
        )

        if end_idx >= len(step_lines):
            break

    return step_docs


def load_gsm8k_docs(
    split="train",
    limit=None,
    chunk_mode=CHUNK_MODE_FULL,
    step_window_size=DEFAULT_STEP_WINDOW_SIZE,
    step_overlap=DEFAULT_STEP_OVERLAP,
    include_full_in_hybrid=True,
):
    """
    Load the GSM8K dataset and classify into PSLE topics.
    
    This dataset contains 8.5K high quality grade school math word problems
    with step-by-step solutions. Each question is automatically classified
    into one of 6 PSLE math topics using keyword-based classification.
    
    Args:
        split: "train" or "test"
        limit: Maximum number of documents to load (None = load all)
        chunk_mode: "full", "step", or "hybrid"
        step_window_size: Number of steps per step-chunk window
        step_overlap: Overlap between adjacent step windows
        include_full_in_hybrid: Whether hybrid mode includes full-problem chunks
    
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
    
    if chunk_mode not in VALID_CHUNK_MODES:
        raise ValueError(
            f"Invalid chunk_mode='{chunk_mode}'. "
            f"Choose one of: {sorted(VALID_CHUNK_MODES)}"
        )

    if step_window_size <= 0:
        raise ValueError("step_window_size must be >= 1")
    if step_overlap < 0:
        raise ValueError("step_overlap must be >= 0")
    if step_overlap >= step_window_size:
        raise ValueError("step_overlap must be smaller than step_window_size")

    docs = []
    topic_counts = {}
    invalid_rows = 0
    invalid_reasons = {
        "missing_question": 0,
        "missing_answer": 0,
        "empty_solution": 0,
    }
    chunk_type_counts = {"full": 0, "solution_step": 0}
    
    max_docs = len(dataset) if limit is None else min(limit, len(dataset))
    
    for i in range(max_docs):
        row = dataset[i]

        raw_question = row.get("question", "")
        raw_answer = row.get("answer", "")

        question = _clean_text(raw_question)
        full_answer = _clean_text(raw_answer)

        # Validation: skip malformed rows to keep index quality stable.
        if not question:
            invalid_rows += 1
            invalid_reasons["missing_question"] += 1
            continue
        if not full_answer:
            invalid_rows += 1
            invalid_reasons["missing_answer"] += 1
            continue

        step_lines, solution_text, final_answer = _extract_solution_parts(full_answer)
        if not solution_text:
            invalid_rows += 1
            invalid_reasons["empty_solution"] += 1
            continue
        
        # Classify question into PSLE topic
        topic = classify_question(question)
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

        if chunk_mode == CHUNK_MODE_FULL or (chunk_mode == CHUNK_MODE_HYBRID and include_full_in_hybrid):
            docs.append(
                _make_full_doc(
                    question=question,
                    solution_text=solution_text,
                    final_answer=final_answer,
                    split=split,
                    row_id=i,
                    topic=topic,
                )
            )
            chunk_type_counts["full"] += 1

        if chunk_mode in {CHUNK_MODE_STEP, CHUNK_MODE_HYBRID}:
            step_docs = _make_step_docs(
                question=question,
                step_lines=step_lines,
                final_answer=final_answer,
                split=split,
                row_id=i,
                topic=topic,
                step_window_size=step_window_size,
                step_overlap=step_overlap,
            )
            docs.extend(step_docs)
            chunk_type_counts["solution_step"] += len(step_docs)
    
    print(f"[ingest] Loaded {len(docs)} documents from GSM8K {split} split.")
    print(
        f"[ingest] Chunk mode='{chunk_mode}' "
        f"(full={chunk_type_counts['full']}, step={chunk_type_counts['solution_step']})."
    )
    if invalid_rows > 0:
        print(
            f"[ingest] Skipped {invalid_rows} invalid rows "
            f"(missing_question={invalid_reasons['missing_question']}, "
            f"missing_answer={invalid_reasons['missing_answer']}, "
            f"empty_solution={invalid_reasons['empty_solution']})."
        )
    print(f"[ingest] Topic distribution:")
    classified_total = sum(topic_counts.values())
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / classified_total * 100.0) if classified_total > 0 else 0.0
        print(f"  - {topic}: {count} ({pct:.1f}%)")
    
    return docs


def get_all_documents(
    use_train=True,
    use_test=False,
    train_limit=None,
    test_limit=None,
    chunk_mode=CHUNK_MODE_FULL,
    step_window_size=DEFAULT_STEP_WINDOW_SIZE,
    step_overlap=DEFAULT_STEP_OVERLAP,
    include_full_in_hybrid=True,
):
    """
    Load documents from GSM8K dataset.
    
    Args:
        use_train: Include training set (7473 examples)
        use_test: Include test set (1319 examples)
        train_limit: Limit training examples (None = all)
        test_limit: Limit test examples (None = all)
        chunk_mode: "full", "step", or "hybrid"
        step_window_size: Number of steps per step-chunk window
        step_overlap: Overlap between adjacent step windows
        include_full_in_hybrid: Whether hybrid mode includes full-problem chunks
    
    Returns:
        Combined list of documents from requested splits
    """
    all_docs = []
    
    if use_train:
        all_docs.extend(
            load_gsm8k_docs(
                split="train",
                limit=train_limit,
                chunk_mode=chunk_mode,
                step_window_size=step_window_size,
                step_overlap=step_overlap,
                include_full_in_hybrid=include_full_in_hybrid,
            )
        )
    
    if use_test:
        all_docs.extend(
            load_gsm8k_docs(
                split="test",
                limit=test_limit,
                chunk_mode=chunk_mode,
                step_window_size=step_window_size,
                step_overlap=step_overlap,
                include_full_in_hybrid=include_full_in_hybrid,
            )
        )
    
    print(f"[ingest] Total documents collected: {len(all_docs)}")
    return all_docs
