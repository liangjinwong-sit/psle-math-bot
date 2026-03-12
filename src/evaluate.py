"""
Evaluation script for the PSLE Math Study Bot.

Runs benchmark questions through the RAG pipeline and measures:
1. Topic classification accuracy
2. Retrieval relevance (do retrieved docs match the expected topic?)
3. Answer correctness (does the final answer match expected?)
4. Citation quality (are citations from the right topic?)

Usage:
    python -m src.evaluate                    # Run full evaluation
    python -m src.evaluate --quick            # Quick mode (skip LLM generation)
    python -m src.evaluate --topic percentage # Evaluate only one topic
"""

import json
import re
import time
import argparse
from typing import List, Dict, Optional

from src.topic_classifier import classify_question, get_topic_display_name
from src.retrieval import retrieve_with_scores, retrieve_by_topic


# ── Benchmark Questions ─────────────────────────────────────────────
# Structured benchmark set covering all 6 PSLE topic families.
# Each question has an expected answer, topic, and method.

BENCHMARK_QUESTIONS = [
    # ── Percentage (6 questions) ──
    {"id": "Q01", "question": "Find 25% of 80.", "expected_answer": "20", "topic": "percentage", "method": "percentage of a quantity"},
    {"id": "Q02", "question": "A shirt costs $60 and is sold at a 20% discount. How much is the discount?", "expected_answer": "12", "topic": "percentage", "method": "percentage discount"},
    {"id": "Q03", "question": "A number increases from 40 to 50. Find the percentage increase.", "expected_answer": "25", "topic": "percentage", "method": "percentage increase"},
    {"id": "Q04", "question": "Convert 35% to a decimal.", "expected_answer": "0.35", "topic": "percentage", "method": "percentage to decimal"},
    {"id": "Q05", "question": "Convert 60% to a fraction in simplest form.", "expected_answer": "3/5", "topic": "percentage", "method": "percentage to fraction"},
    {"id": "Q06", "question": "Find 15% of 200.", "expected_answer": "30", "topic": "percentage", "method": "percentage of a quantity"},
    
    # ── Fractions & Decimals (6 questions) ──
    {"id": "Q07", "question": "Calculate 1/4 + 2/4.", "expected_answer": "3/4", "topic": "fractions_decimals", "method": "adding fractions"},
    {"id": "Q08", "question": "Which is greater, 3/5 or 0.7?", "expected_answer": "0.7", "topic": "fractions_decimals", "method": "comparing fractions and decimals"},
    {"id": "Q09", "question": "Calculate 4.25 - 1.8.", "expected_answer": "2.45", "topic": "fractions_decimals", "method": "decimal subtraction"},
    {"id": "Q10", "question": "Convert 3/10 to a decimal.", "expected_answer": "0.3", "topic": "fractions_decimals", "method": "fraction to decimal"},
    {"id": "Q11", "question": "Convert 0.25 to a fraction in simplest form.", "expected_answer": "1/4", "topic": "fractions_decimals", "method": "decimal to fraction"},
    {"id": "Q12", "question": "Which is smaller, 0.48 or 0.5?", "expected_answer": "0.48", "topic": "fractions_decimals", "method": "comparing decimals"},
    
    # ── Ratio & Proportion (4 questions) ──
    {"id": "Q13", "question": "A recipe uses sugar and flour in the ratio 2:5. If there are 10 cups of flour, how many cups of sugar are needed?", "expected_answer": "4", "topic": "ratio_proportion", "method": "equivalent ratio"},
    {"id": "Q14", "question": "Divide $84 between Ali and Ben in the ratio 3:4. How much does Ali get?", "expected_answer": "36", "topic": "ratio_proportion", "method": "sharing in a ratio"},
    {"id": "Q15", "question": "The ratio of red to blue marbles is 3:5. If there are 24 marbles in total, how many are red?", "expected_answer": "9", "topic": "ratio_proportion", "method": "ratio with total"},
    {"id": "Q16", "question": "Tom and Jerry share stickers in the ratio 2:3. If Jerry has 15 stickers, how many does Tom have?", "expected_answer": "10", "topic": "ratio_proportion", "method": "finding missing ratio value"},
    
    # ── Rate / Unitary Reasoning (4 questions) ──
    {"id": "Q17", "question": "A car travels 180 km in 3 hours. What is its average speed?", "expected_answer": "60", "topic": "rate", "method": "speed = distance / time"},
    {"id": "Q18", "question": "5 notebooks cost $15. How much does 1 notebook cost?", "expected_answer": "3", "topic": "rate", "method": "unitary method"},
    {"id": "Q19", "question": "A printer prints 120 pages in 4 minutes. How many pages does it print per minute?", "expected_answer": "30", "topic": "rate", "method": "rate per unit"},
    {"id": "Q20", "question": "If 8 workers can build a wall in 6 days, how many days would 12 workers take?", "expected_answer": "4", "topic": "rate", "method": "inverse proportion"},
    
    # ── Measurement (4 questions) ──
    {"id": "Q21", "question": "A rectangle has length 8 cm and width 5 cm. Find its area.", "expected_answer": "40", "topic": "measurement", "method": "area of rectangle"},
    {"id": "Q22", "question": "A rectangle has length 9 cm and width 4 cm. Find its perimeter.", "expected_answer": "26", "topic": "measurement", "method": "perimeter of rectangle"},
    {"id": "Q23", "question": "Find the area of a triangle with base 10 cm and height 6 cm.", "expected_answer": "30", "topic": "measurement", "method": "area of triangle"},
    {"id": "Q24", "question": "A cube has side length 3 cm. Find its volume.", "expected_answer": "27", "topic": "measurement", "method": "volume of cube"},
    
    # ── Data Handling (4 questions) ──
    {"id": "Q25", "question": "Find the mean of 6, 8, 10, and 12.", "expected_answer": "9", "topic": "data_handling", "method": "mean / average"},
    {"id": "Q26", "question": "A student scored 12, 15, and 18 in three tests. What is the average score?", "expected_answer": "15", "topic": "data_handling", "method": "mean / average"},
    {"id": "Q27", "question": "The average of 5 numbers is 20. What is their total sum?", "expected_answer": "100", "topic": "data_handling", "method": "sum from average"},
    {"id": "Q28", "question": "The mean of 4 numbers is 10. If three of the numbers are 8, 12, and 11, find the fourth number.", "expected_answer": "9", "topic": "data_handling", "method": "finding missing value from mean"},
]


def extract_numeric(text: str) -> Optional[str]:
    """
    Extract the core numeric value from an answer string for comparison.
    Handles formats like: "20", "$12", "25%", "3/5", "0.35", "40 cm²"
    
    Returns a cleaned string for comparison, or None if no number found.
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Try to find a fraction first (e.g., "3/5", "1/4")
    frac_match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if frac_match:
        return f"{frac_match.group(1)}/{frac_match.group(2)}"
    
    # Find decimal or integer numbers
    num_match = re.search(r"-?\d+\.?\d*", text)
    if num_match:
        return num_match.group(0)
    
    return None


def answers_match(expected: str, actual: str) -> bool:
    """
    Check if the actual answer matches the expected answer.
    Handles numeric comparison, fraction matching, and common formats.
    """
    exp_num = extract_numeric(expected)
    act_num = extract_numeric(actual)
    
    if exp_num is None or act_num is None:
        return False
    
    # Direct string match
    if exp_num == act_num:
        return True
    
    # Try numeric comparison (handles "20" vs "20.0")
    try:
        # Handle fractions
        if "/" in exp_num:
            parts = exp_num.split("/")
            exp_val = float(parts[0]) / float(parts[1])
        else:
            exp_val = float(exp_num)
        
        if "/" in act_num:
            parts = act_num.split("/")
            act_val = float(parts[0]) / float(parts[1])
        else:
            act_val = float(act_num)
        
        return abs(exp_val - act_val) < 0.01
    except (ValueError, ZeroDivisionError):
        return False


def evaluate_topic_classification(questions: List[Dict]) -> Dict:
    """
    Evaluate how accurately the topic classifier assigns questions to PSLE topics.
    
    Returns:
        Dict with accuracy, per-topic results, and confusion details
    """
    print("\n" + "=" * 60)
    print("EVALUATION 1: Topic Classification Accuracy")
    print("=" * 60)
    
    correct = 0
    total = len(questions)
    topic_results = {}
    
    for q in questions:
        predicted = classify_question(q["question"])
        expected = q["topic"]
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
        
        # Track per-topic accuracy
        if expected not in topic_results:
            topic_results[expected] = {"correct": 0, "total": 0, "errors": []}
        topic_results[expected]["total"] += 1
        if is_correct:
            topic_results[expected]["correct"] += 1
        else:
            topic_results[expected]["errors"].append({
                "id": q["id"],
                "question": q["question"][:60],
                "predicted": predicted,
            })
        
        status = "✅" if is_correct else "❌"
        print(f"  {status} {q['id']}: expected={expected}, got={predicted}")
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n  Overall: {correct}/{total} ({accuracy * 100:.1f}%)")
    print(f"\n  Per-topic breakdown:")
    for topic, data in sorted(topic_results.items()):
        t_acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        print(f"    {get_topic_display_name(topic)}: {data['correct']}/{data['total']} ({t_acc * 100:.0f}%)")
        for err in data["errors"]:
            print(f"      ❌ {err['id']}: predicted '{err['predicted']}' — \"{err['question']}...\"")
    
    return {"accuracy": accuracy, "correct": correct, "total": total, "per_topic": topic_results}


def evaluate_retrieval_relevance(questions: List[Dict], k: int = 4) -> Dict:
    """
    Evaluate whether retrieved documents match the expected topic.
    
    Measures:
    - Topic precision: fraction of retrieved docs from the correct topic
    - Average similarity score of retrieved docs
    
    Returns:
        Dict with topic_precision, avg_score, and per-question details
    """
    print("\n" + "=" * 60)
    print("EVALUATION 2: Retrieval Relevance")
    print("=" * 60)
    
    total_docs = 0
    topic_match_docs = 0
    all_scores = []
    details = []
    
    for q in questions:
        results = retrieve_with_scores(q["question"], k=k)
        
        q_topic_matches = 0
        q_scores = []
        
        for doc, score in results:
            total_docs += 1
            q_scores.append(score)
            all_scores.append(score)
            
            if doc.metadata.get("topic") == q["topic"]:
                q_topic_matches += 1
                topic_match_docs += 1
        
        precision = q_topic_matches / len(results) if results else 0
        avg_score = sum(q_scores) / len(q_scores) if q_scores else 0
        
        status = "✅" if precision >= 0.5 else "⚠️" if precision > 0 else "❌"
        print(f"  {status} {q['id']} [{q['topic']}]: {q_topic_matches}/{len(results)} on-topic, avg_sim={avg_score:.3f}")
        
        details.append({
            "id": q["id"],
            "topic_precision": precision,
            "avg_score": avg_score,
            "num_retrieved": len(results),
        })
    
    overall_precision = topic_match_docs / total_docs if total_docs > 0 else 0
    overall_avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    print(f"\n  Overall topic precision: {topic_match_docs}/{total_docs} ({overall_precision * 100:.1f}%)")
    print(f"  Overall avg similarity: {overall_avg_score:.3f}")
    
    return {
        "topic_precision": overall_precision,
        "avg_similarity": overall_avg_score,
        "total_docs": total_docs,
        "topic_match_docs": topic_match_docs,
        "details": details,
    }


def evaluate_answer_correctness(questions: List[Dict], k: int = 4) -> Dict:
    """
    Evaluate end-to-end answer correctness using the full RAG + LLM pipeline.
    
    Sends each benchmark question through the pipeline and checks if the
    generated answer contains the expected numeric answer.
    
    NOTE: Requires a valid GOOGLE_API_KEY and makes LLM API calls.
    
    Returns:
        Dict with accuracy, per-question results, and timing info
    """
    from src.generation import answer_question
    
    print("\n" + "=" * 60)
    print("EVALUATION 3: Answer Correctness (end-to-end)")
    print("=" * 60)
    
    correct = 0
    total = len(questions)
    details = []
    total_time = 0
    
    for i, q in enumerate(questions):
        print(f"  [{i + 1}/{total}] {q['id']}: {q['question'][:50]}...", end=" ", flush=True)
        
        start = time.time()
        try:
            result = answer_question(q["question"], topic=q["topic"], k=k)
            elapsed = time.time() - start
            total_time += elapsed
            
            answer_text = result["answer"]
            is_correct = answers_match(q["expected_answer"], answer_text)
            
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} ({elapsed:.1f}s) expected={q['expected_answer']}")
            
            details.append({
                "id": q["id"],
                "correct": is_correct,
                "expected": q["expected_answer"],
                "confidence": result.get("confidence", 0),
                "time_seconds": elapsed,
            })
        
        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"❌ ERROR: {str(e)[:60]}")
            details.append({
                "id": q["id"],
                "correct": False,
                "expected": q["expected_answer"],
                "error": str(e)[:100],
                "time_seconds": elapsed,
            })
    
    accuracy = correct / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    
    print(f"\n  Accuracy: {correct}/{total} ({accuracy * 100:.1f}%)")
    print(f"  Avg time per question: {avg_time:.1f}s")
    print(f"  Total evaluation time: {total_time:.1f}s")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time_seconds": avg_time,
        "total_time_seconds": total_time,
        "details": details,
    }


def run_evaluation(quick: bool = False, topic_filter: str = None):
    """
    Run the full evaluation suite.
    
    Args:
        quick: If True, skip LLM-based answer correctness (eval 3)
        topic_filter: If set, only evaluate questions from this topic
    """
    questions = BENCHMARK_QUESTIONS
    
    if topic_filter:
        questions = [q for q in questions if q["topic"] == topic_filter]
        if not questions:
            print(f"No benchmark questions found for topic: {topic_filter}")
            return
        print(f"Filtering to topic: {get_topic_display_name(topic_filter)} ({len(questions)} questions)")
    
    print(f"\nRunning evaluation on {len(questions)} benchmark questions...")
    
    # Eval 1: Topic classification (no API calls needed)
    classification_results = evaluate_topic_classification(questions)
    
    # Eval 2: Retrieval relevance (no API calls needed, but needs FAISS index)
    try:
        retrieval_results = evaluate_retrieval_relevance(questions)
    except Exception as e:
        print(f"\n  ❌ Retrieval evaluation failed: {e}")
        print("     Make sure you've built the index: python build_index.py")
        retrieval_results = None
    
    # Eval 3: Answer correctness (needs API key + LLM calls)
    answer_results = None
    if not quick:
        try:
            answer_results = evaluate_answer_correctness(questions)
        except Exception as e:
            print(f"\n  ❌ Answer evaluation failed: {e}")
            print("     Make sure GOOGLE_API_KEY is set in .env")
    else:
        print("\n  ⏭️  Skipping answer correctness (quick mode). Run without --quick for full eval.")
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Benchmark size: {len(questions)} questions")
    print(f"  Topic classification accuracy: {classification_results['accuracy'] * 100:.1f}%")
    if retrieval_results:
        print(f"  Retrieval topic precision: {retrieval_results['topic_precision'] * 100:.1f}%")
        print(f"  Retrieval avg similarity: {retrieval_results['avg_similarity']:.3f}")
    if answer_results:
        print(f"  Answer correctness: {answer_results['accuracy'] * 100:.1f}%")
        print(f"  Avg response time: {answer_results['avg_time_seconds']:.1f}s")
    print("=" * 60)
    
    # Save results to file
    results = {
        "num_questions": len(questions),
        "topic_filter": topic_filter,
        "classification": classification_results,
        "retrieval": retrieval_results,
        "answer": answer_results,
    }
    
    output_path = "data/benchmark/evaluation_results.json"
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")
    except Exception as e:
        print(f"\n  Warning: Could not save results to file: {e}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the PSLE Math Study Bot")
    parser.add_argument("--quick", action="store_true", help="Skip LLM-based evaluation (faster, no API calls)")
    parser.add_argument("--topic", type=str, default=None, help="Evaluate only this topic (e.g., 'percentage', 'rate')")
    args = parser.parse_args()
    
    run_evaluation(quick=args.quick, topic_filter=args.topic)
