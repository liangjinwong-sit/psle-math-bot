"""
Practice mode: random questions from GSM8K test set and LLM-generated questions.

Features:
- Pull random practice questions from GSM8K test split
- Filter by PSLE topic for targeted practice
- Generate new practice questions using LLM (extension feature)
"""

import random
from datasets import load_dataset
from src.topic_classifier import classify_question, get_topic_display_name

# ── Difficulty Thresholds (step counts) ────────────────────────────
# GSM8K solutions are one line per reasoning step.  We classify by
# counting non-empty, non-answer lines (steps).
DIFFICULTY_EASY_MAX_STEPS = 2    # <= 2 steps → easy
DIFFICULTY_MEDIUM_MAX_STEPS = 4  # 3-4 steps → medium; >4 → hard

_test_dataset = None
_classified_test = None  # Cache of (index, topic) pairs for filtered retrieval


def _load_test_split():
    """Lazily load the GSM8K test split and cache it."""
    global _test_dataset
    if _test_dataset is None:
        _test_dataset = load_dataset("openai/gsm8k", "main", split="test")
        print(f"[practice] Loaded GSM8K test split ({len(_test_dataset)} questions).")
    return _test_dataset


def _get_classified_test():
    """Lazily classify all test questions by topic and cache the results."""
    global _classified_test
    if _classified_test is None:
        dataset = _load_test_split()
        _classified_test = {}
        for i in range(len(dataset)):
            topic = classify_question(dataset[i]["question"])
            if topic not in _classified_test:
                _classified_test[topic] = []
            _classified_test[topic].append(i)
        print(f"[practice] Classified test set by topic:")
        for topic, indices in sorted(_classified_test.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  - {topic}: {len(indices)} questions")
    return _classified_test


def get_random_question(topic: str = None, difficulty: str = None):
    """
    Return a random practice question, optionally filtered by topic and difficulty.

    Args:
        topic: Optional PSLE topic key (e.g., "percentage", "rate").
               If None, returns a question from any topic.
        difficulty: Optional difficulty level ("easy", "medium", "hard").
                    If None, returns any difficulty.

    Returns:
        dict with question, answer, topic, and difficulty
    """
    dataset = _load_test_split()

    if topic:
        classified = _get_classified_test()
        indices = classified.get(topic, [])
        if not indices:
            indices = list(range(len(dataset)))
    else:
        indices = list(range(len(dataset)))

    # Filter by difficulty if specified
    if difficulty:
        filtered = [i for i in indices
                    if estimate_difficulty(dataset[i]["answer"]) == difficulty]
        if filtered:
            indices = filtered

    idx = random.choice(indices)
    row = dataset[idx]
    question_topic = classify_question(row["question"])

    return {
        "question": row["question"],
        "answer": row["answer"],
        "topic": question_topic,
        "topic_display": get_topic_display_name(question_topic),
        "difficulty": estimate_difficulty(row["answer"]),
    }


def get_final_answer(full_answer: str):
    """Extract only the numeric answer after '####'.
    Returns the full string if '####' is not present.
    """
    if "####" in full_answer:
        return full_answer.split("####")[-1].strip()
    return full_answer.strip()


def estimate_difficulty(answer_text: str) -> str:
    """
    Estimate question difficulty from GSM8K solution step count.

    Easy: 1-2 reasoning steps
    Medium: 3-4 reasoning steps
    Hard: 5+ reasoning steps
    """
    # Heuristic: count reasoning steps (non-empty, non-answer lines).
    # GSM8K solutions use one line per step, so step count correlates with difficulty.
    steps = [line.strip() for line in answer_text.split("\n")
             if line.strip() and "####" not in line]
    if len(steps) <= DIFFICULTY_EASY_MAX_STEPS:
        return "easy"      # 1-2 steps: straightforward
    elif len(steps) <= DIFFICULTY_MEDIUM_MAX_STEPS:
        return "medium"    # 3-4 steps: moderate complexity
    else:
        return "hard"      # 5+ steps: multi-step reasoning


def generate_practice_question(topic: str, difficulty: str = "medium"):
    """
    Use the LLM to generate a new practice question for a given topic.
    
    This is the extension feature: instead of just pulling from GSM8K,
    we generate fresh questions so students get unlimited practice.
    
    Args:
        topic: PSLE topic key (e.g., "percentage", "rate")
        difficulty: One of "easy", "medium", "hard"
    
    Returns:
        dict with generated question, solution, answer, and metadata
    """
    from src.generation import _get_llm
    from src.retrieval import retrieve_by_topic
    
    topic_display = get_topic_display_name(topic)
    
    # Retrieve a few examples from this topic to use as style reference
    try:
        example_docs = retrieve_by_topic(f"{topic_display} word problem", topic, k=3)
        example_text = ""
        for i, (doc, score) in enumerate(example_docs, 1):
            q = doc.metadata.get("question", "")
            example_text += f"Example {i}: {q}\n"
    except Exception:
        example_text = "No examples available."
    
    # Difficulty settings
    difficulty_guide = {
        "easy": "Use small, simple numbers (single digit or small two-digit). One-step problem.",
        "medium": "Use reasonable numbers. Two-step problem requiring one main concept.",
        "hard": "Use larger or trickier numbers. Multi-step problem combining concepts.",
    }
    
    prompt_text = f"""You are a PSLE Math question writer for Primary 5-6 students in Singapore.

Topic: {topic_display}
Difficulty: {difficulty}
Difficulty guide: {difficulty_guide.get(difficulty, difficulty_guide["medium"])}

Here are some example questions from this topic for reference:
{example_text}

Generate ONE new original math word problem for this topic.
Use Singapore context where possible (names like Ali, Mei Ling, Raju; SGD currency; local settings).

Respond in EXACTLY this format (no extra text):
QUESTION: [Your question here]
SOLUTION: [Step-by-step solution]
ANSWER: [Final numeric answer only]"""

    try:
        llm = _get_llm()
        result = llm.invoke(prompt_text)
        response = result.content.strip()
        
        # Parse the structured response
        question = ""
        solution = ""
        answer = ""
        
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("QUESTION:"):
                question = line[len("QUESTION:"):].strip()
            elif line.startswith("SOLUTION:"):
                solution = line[len("SOLUTION:"):].strip()
            elif line.startswith("ANSWER:"):
                answer = line[len("ANSWER:"):].strip()
        
        # If structured parsing didn't work well, try to extract from full response
        if not question:
            parts = response.split("SOLUTION:")
            if len(parts) >= 2:
                question = parts[0].replace("QUESTION:", "").strip()
                rest = parts[1]
                answer_parts = rest.split("ANSWER:")
                solution = answer_parts[0].strip()
                if len(answer_parts) >= 2:
                    answer = answer_parts[1].strip()
        
        # Handle multi-line solution: extract everything between SOLUTION:
        # and ANSWER: markers when line-by-line parsing missed content.
        if not solution and "SOLUTION:" in response and "ANSWER:" in response:
            sol_start = response.index("SOLUTION:") + len("SOLUTION:")
            sol_end = response.index("ANSWER:")
            solution = response[sol_start:sol_end].strip()
            answer = response[sol_end + len("ANSWER:"):].strip()
        
        return {
            "question": question or "Error: Could not parse generated question.",
            "solution": solution or "Solution not available.",
            "answer": answer or "N/A",
            "topic": topic,
            "topic_display": topic_display,
            "difficulty": difficulty,
            "generated": True,
        }
    
    except Exception as e:
        return {
            "question": f"Error generating question: {str(e)}",
            "solution": "",
            "answer": "",
            "topic": topic,
            "topic_display": topic_display,
            "difficulty": difficulty,
            "generated": False,
        }
