import random
from datasets import load_dataset

_test_dataset = None


def _load_test_split():
    """Lazily load the GSM8K test split and cache it."""
    global _test_dataset
    if _test_dataset is None:
        _test_dataset = load_dataset("openai/gsm8k", "main", split="test")
        print(f"[practice] Loaded GSM8K test split ({len(_test_dataset)} questions).")
    return _test_dataset


def get_random_question():
    """Return a random practice question from the GSM8K test split."""
    dataset = _load_test_split()
    row = dataset[random.randint(0, len(dataset) - 1)]
    return {
        "question": row["question"],
        "answer": row["answer"],
    }


def get_final_answer(full_answer: str):
    """Extract only the numeric answer after '####'. Returns the full string if '####' is not present."""
    if "####" in full_answer:
        return full_answer.split("####")[-1].strip()
    return full_answer.strip()
