"""
LLM Provider Comparison Script for PSLE Math Study Bot.

Produces two outputs that your team can screenshot for the presentation:
1. Side-by-side answer comparison — same question, 4 different LLMs
2. Evaluation summary table — accuracy, speed, cost across providers

Usage:
    python compare_providers.py                    # Full comparison (answers + eval table)
    python compare_providers.py --answers-only     # Just the side-by-side answers
    python compare_providers.py --table-only       # Just the eval summary table
"""

import json
import glob
import os
import time
import argparse
from dotenv import load_dotenv

load_dotenv()


# ── Test questions for side-by-side comparison ──────────────────────
# Pick one from each topic to show variety in the demo.
COMPARISON_QUESTIONS = [
    {
        "question": "A bag originally costs $80. After a 25% discount, what is the sale price?",
        "topic": "percentage",
        "expected": "60",
    },
    {
        "question": "The ratio of boys to girls in a class is 4:5. If there are 20 girls, how many students are there altogether?",
        "topic": "ratio_proportion",
        "expected": "36",
    },
    {
        "question": "A machine produces 480 items in 8 hours. How many items does it produce in 5 hours?",
        "topic": "rate",
        "expected": "300",
    },
]

# Provider display info
PROVIDER_INFO = {
    "gemini": {"name": "Gemini 2.5 Flash", "type": "Proprietary (Cloud)", "cost": "Free tier"},
    "openai": {"name": "GPT-4o-mini", "type": "Proprietary (Cloud)", "cost": "Paid"},
    "groq":   {"name": "Llama 3.1 8B (Groq)", "type": "Open-source (Cloud)", "cost": "Free"},
    "ollama": {"name": "Llama 3.2 3B (Local)", "type": "Open-source (Local)", "cost": "Free"},
}


def check_provider_available(provider: str) -> bool:
    """Check if a provider's API key is configured."""
    if provider == "gemini":
        key = os.getenv("GOOGLE_API_KEY", "")
        return bool(key) and key != "your_google_api_key_here"
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        return bool(key) and key != "your_openai_api_key_here"
    elif provider == "groq":
        key = os.getenv("GROQ_API_KEY", "")
        return bool(key) and key != "your_groq_api_key_here"
    elif provider == "ollama":
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False
    return False


def run_answer_comparison():
    """Run the same questions through each available provider and print side-by-side."""
    from src.generation import answer_question, switch_provider, get_current_provider

    available = []
    unavailable = []
    for p in ["gemini", "openai", "groq", "ollama"]:
        if check_provider_available(p):
            available.append(p)
        else:
            unavailable.append(p)

    if not available:
        print("ERROR: No LLM providers are configured. Check your .env file.")
        return

    print()
    print("=" * 80)
    print("  LLM PROVIDER COMPARISON -- Side-by-Side Answers")
    print("=" * 80)
    print(f"  Available providers: {', '.join(available)}")
    if unavailable:
        print(f"  Skipped (not configured): {', '.join(unavailable)}")
    print()

    for q_info in COMPARISON_QUESTIONS:
        question = q_info["question"]
        topic = q_info["topic"]
        expected = q_info["expected"]

        print("-" * 80)
        print(f"  QUESTION: {question}")
        print(f"  Topic: {topic}  |  Expected answer: {expected}")
        print("-" * 80)

        for provider in available:
            switch_provider(provider)
            info = PROVIDER_INFO[provider]

            print(f"\n  [{info['name']}] ({info['type']}, {info['cost']})")
            print(f"  {'.' * 60}")

            start = time.time()
            try:
                result = answer_question(question, topic=topic)
                elapsed = time.time() - start

                answer_lines = result["answer"].strip().split("\n")
                preview = "\n".join(f"    {line}" for line in answer_lines[:8])
                if len(answer_lines) > 8:
                    preview += f"\n    ... ({len(answer_lines) - 8} more lines)"

                print(preview)
                print(f"\n    Time: {elapsed:.1f}s | Confidence: {result['confidence']:.0%} | Tools: {result['used_tools'] or 'none'}")

            except Exception as e:
                elapsed = time.time() - start
                print(f"    ERROR: {str(e)[:80]}")
                print(f"    Time: {elapsed:.1f}s")

        print()

    print("=" * 80)
    print("  END OF COMPARISON")
    print("=" * 80)


def print_eval_table():
    """Load eval_*.json files and print a formatted comparison table."""
    eval_files = sorted(glob.glob("data/benchmark/eval_*.json"))

    if not eval_files:
        print()
        print("ERROR: No eval files found in data/benchmark/eval_*.json")
        print()
        print("Run these commands first:")
        print("  python -m src.evaluate --provider gemini")
        print("  copy data\\benchmark\\evaluation_results.json data\\benchmark\\eval_gemini.json")
        print()
        print("  python -m src.evaluate --provider openai")
        print("  copy data\\benchmark\\evaluation_results.json data\\benchmark\\eval_openai.json")
        print()
        print("  (repeat for groq and ollama)")
        return

    results = {}
    for path in eval_files:
        with open(path) as f:
            data = json.load(f)
        provider = data.get("provider", os.path.basename(path).replace("eval_", "").replace(".json", ""))
        results[provider] = data

    providers = list(results.keys())

    print()
    print("=" * 80)
    print("  EVALUATION RESULTS -- Multi-Provider Comparison")
    print("  Benchmark: 56 questions across 6 PSLE topics")
    print("=" * 80)
    print()

    col_w = 18
    header = f"  {'Metric':<32}"
    for p in providers:
        info = PROVIDER_INFO.get(p, {"name": p})
        header += f" {info['name']:>{col_w}}"
    print(header)
    print("  " + "-" * (32 + (col_w + 1) * len(providers)))

    # Classification
    row = f"  {'Classification accuracy':<32}"
    for p in providers:
        val = results[p]["classification"]["accuracy"]
        row += f" {val*100:>{col_w-1}.1f}%"
    print(row)

    # Retrieval
    row = f"  {'Retrieval topic precision':<32}"
    for p in providers:
        data = results[p].get("retrieval")
        if data:
            row += f" {data['topic_precision']*100:>{col_w-1}.1f}%"
        else:
            row += f" {'N/A':>{col_w}}"
    print(row)

    row = f"  {'Retrieval avg similarity':<32}"
    for p in providers:
        data = results[p].get("retrieval")
        if data:
            row += f" {data['avg_similarity']:>{col_w}.3f}"
        else:
            row += f" {'N/A':>{col_w}}"
    print(row)

    # Answer correctness
    row = f"  {'Answer correctness':<32}"
    for p in providers:
        data = results[p].get("answer")
        if data:
            row += f" {data['accuracy']*100:>{col_w-1}.1f}%"
        else:
            row += f" {'(run full eval)':>{col_w}}"
    print(row)

    row = f"  {'Avg response time (s)':<32}"
    for p in providers:
        data = results[p].get("answer")
        if data:
            row += f" {data['avg_time_seconds']:>{col_w}.1f}"
        else:
            row += f" {'--':>{col_w}}"
    print(row)

    # Provider info
    print()
    print("  " + "-" * (32 + (col_w + 1) * len(providers)))

    row = f"  {'Model':<32}"
    for p in providers:
        info = PROVIDER_INFO.get(p, {"name": p})
        row += f" {info['name']:>{col_w}}"
    print(row)

    row = f"  {'Type':<32}"
    for p in providers:
        info = PROVIDER_INFO.get(p, {"type": "?"})
        row += f" {info['type']:>{col_w}}"
    print(row)

    row = f"  {'Cost':<32}"
    for p in providers:
        info = PROVIDER_INFO.get(p, {"cost": "?"})
        row += f" {info['cost']:>{col_w}}"
    print(row)

    print()
    print("  Notes:")
    print("  - Classification and retrieval use LOCAL embeddings (identical across all providers)")
    print("  - Only answer generation uses the LLM (this is where providers differ)")
    if "groq" in providers and "ollama" in providers:
        print("  - Groq vs Ollama: same Llama 3.1 8B weights, different infra (cloud vs local)")
    print()
    print("=" * 80)

    # Per-provider wrong answers
    has_answer_data = any(results[p].get("answer") for p in providers)
    if has_answer_data:
        print()
        print("  PER-PROVIDER DETAIL")
        print("  " + "-" * 60)

        for p in providers:
            ans = results[p].get("answer")
            if not ans:
                continue
            correct_ids = [d["id"] for d in ans["details"] if d.get("correct")]
            wrong_ids = [d["id"] for d in ans["details"] if not d.get("correct") and "error" not in d]
            error_ids = [d["id"] for d in ans["details"] if "error" in d]
            info = PROVIDER_INFO.get(p, {"name": p})
            print(f"\n  {info['name']}: {ans['correct']}/{ans['total']} correct ({ans['accuracy']*100:.1f}%)")
            if wrong_ids:
                print(f"    Wrong: {', '.join(wrong_ids)}")
            if error_ids:
                print(f"    Errors: {', '.join(error_ids)}")

        print()
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LLM providers for PSLE Math Study Bot")
    parser.add_argument("--answers-only", action="store_true", help="Only run side-by-side answer comparison")
    parser.add_argument("--table-only", action="store_true", help="Only show evaluation summary table")
    args = parser.parse_args()

    if args.table_only:
        print_eval_table()
    elif args.answers_only:
        run_answer_comparison()
    else:
        print_eval_table()
        run_answer_comparison()