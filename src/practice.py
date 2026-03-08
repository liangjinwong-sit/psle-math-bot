import random
from src.retrieval import load_index

_cached_examples = None


def _load_examples():
    global _cached_examples

    if _cached_examples is not None:
        return _cached_examples

    vectorstore = load_index()
    docs = vectorstore.similarity_search("worked example", k=50)

    cleaned = []
    for doc in docs:
        text = doc.page_content.strip()
        if "Question:" in text:
            cleaned.append(
                {
                    "question": _extract_question(text),
                    "answer": text,
                    "source": doc.metadata.get("source", "unknown"),
                }
            )

    _cached_examples = cleaned
    print(f"[practice] Loaded {len(_cached_examples)} local practice examples from indexed notes.")
    return _cached_examples


def _extract_question(text: str):
    marker = "Question:"
    if marker not in text:
        return text[:200].strip()

    start = text.find(marker) + len(marker)
    remainder = text[start:].strip()

    stop_markers = ["Step 1:", "Answer:", "Why this works:"]
    stop_positions = [remainder.find(m) for m in stop_markers if m in remainder]

    if stop_positions:
        stop = min([p for p in stop_positions if p >= 0])
        return remainder[:stop].strip()

    return remainder.strip()


def get_random_question():
    examples = _load_examples()

    if not examples:
        return {
            "question": "No local practice questions found in the indexed notes.",
            "answer": "Please add more worked examples to your topic markdown files and rebuild the index.",
        }

    row = random.choice(examples)
    return {
        "question": row["question"],
        "answer": row["answer"],
    }


def get_final_answer(full_answer: str):
    if "Answer:" in full_answer:
        after = full_answer.split("Answer:", 1)[1]
        if "Why this works:" in after:
            return after.split("Why this works:", 1)[0].strip()
        return after.strip()
    return full_answer.strip()
