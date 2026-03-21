"""
Evaluation script for the PSLE Math Study Bot.

Runs benchmark questions through the RAG pipeline and measures:
1. Topic classification accuracy
2. Retrieval relevance (do retrieved docs match the expected topic?)
3. Answer correctness (does the final answer match expected?)
4. Citation quality (are citations from the right topic?)

Usage:
    python -m src.evaluate                    # Run full evaluation (default: gemini)
    python -m src.evaluate --quick            # Quick mode (skip LLM generation)
    python -m src.evaluate --topic percentage # Evaluate only one topic
    python -m src.evaluate --provider openai  # Use a different LLM provider
    python -m src.evaluate --provider groq    # Use Groq (Llama 3.1)
    python -m src.evaluate --provider ollama  # Use local Ollama
"""

import csv
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
    # ── Percentage (10 questions) ──
    {"id": "Q01", "question": "Find 25% of 80.", "expected_answer": "20", "topic": "percentage", "method": "percentage of a quantity"},
    {"id": "Q02", "question": "A shirt costs $60 and is sold at a 20% discount. How much is the discount?", "expected_answer": "12", "topic": "percentage", "method": "percentage discount"},
    {"id": "Q03", "question": "A number increases from 40 to 50. Find the percentage increase.", "expected_answer": "25", "topic": "percentage", "method": "percentage increase"},
    {"id": "Q04", "question": "Convert 35% to a decimal.", "expected_answer": "0.35", "topic": "percentage", "method": "percentage to decimal"},
    {"id": "Q05", "question": "Convert 60% to a fraction in simplest form.", "expected_answer": "3/5", "topic": "percentage", "method": "percentage to fraction"},
    {"id": "Q06", "question": "Find 15% of 200.", "expected_answer": "30", "topic": "percentage", "method": "percentage of a quantity"},
    {"id": "Q07", "question": "A bag originally costs $80. After a 25% discount, what is the sale price?", "expected_answer": "60", "topic": "percentage", "method": "percentage discount (find sale price)"},
    {"id": "Q08", "question": "Ali scored 36 out of 40 on a test. What is his score as a percentage?", "expected_answer": "90", "topic": "percentage", "method": "express as percentage"},
    {"id": "Q09", "question": "A population decreased from 500 to 450. Find the percentage decrease.", "expected_answer": "10", "topic": "percentage", "method": "percentage decrease"},
    {"id": "Q10", "question": "Mei Ling saved $150 last month. This month she saved 20% more. How much did she save this month?", "expected_answer": "180", "topic": "percentage", "method": "percentage increase (find new value)"},

    # ── Fractions & Decimals (10 questions) ──
    {"id": "Q11", "question": "Calculate 1/4 + 2/4.", "expected_answer": "3/4", "topic": "fractions_decimals", "method": "adding fractions"},
    {"id": "Q12", "question": "Which is greater, 3/5 or 0.7?", "expected_answer": "0.7", "topic": "fractions_decimals", "method": "comparing fractions and decimals"},
    {"id": "Q13", "question": "Calculate 4.25 - 1.8.", "expected_answer": "2.45", "topic": "fractions_decimals", "method": "decimal subtraction"},
    {"id": "Q14", "question": "Convert 3/10 to a decimal.", "expected_answer": "0.3", "topic": "fractions_decimals", "method": "fraction to decimal"},
    {"id": "Q15", "question": "Convert 0.25 to a fraction in simplest form.", "expected_answer": "1/4", "topic": "fractions_decimals", "method": "decimal to fraction"},
    {"id": "Q16", "question": "Which is smaller, 0.48 or 0.5?", "expected_answer": "0.48", "topic": "fractions_decimals", "method": "comparing decimals"},
    {"id": "Q17", "question": "Calculate 2/3 + 1/6.", "expected_answer": "5/6", "topic": "fractions_decimals", "method": "adding unlike fractions"},
    {"id": "Q18", "question": "Calculate 3/4 x 2/5.", "expected_answer": "3/10", "topic": "fractions_decimals", "method": "multiplying fractions"},
    {"id": "Q19", "question": "Raju ate 2/5 of a pizza. Mei Ling ate 1/3 of the same pizza. What fraction did they eat altogether?", "expected_answer": "11/15", "topic": "fractions_decimals", "method": "adding unlike fractions (word problem)"},
    {"id": "Q20", "question": "Calculate 5.6 + 3.75.", "expected_answer": "9.35", "topic": "fractions_decimals", "method": "decimal addition"},

    # ── Ratio & Proportion (9 questions) ──
    {"id": "Q21", "question": "A recipe uses sugar and flour in the ratio 2:5. If there are 10 cups of flour, how many cups of sugar are needed?", "expected_answer": "4", "topic": "ratio_proportion", "method": "equivalent ratio"},
    {"id": "Q22", "question": "Divide $84 between Ali and Ben in the ratio 3:4. How much does Ali get?", "expected_answer": "36", "topic": "ratio_proportion", "method": "sharing in a ratio"},
    {"id": "Q23", "question": "The ratio of red to blue marbles is 3:5. If there are 24 marbles in total, how many are red?", "expected_answer": "9", "topic": "ratio_proportion", "method": "ratio with total"},
    {"id": "Q24", "question": "Tom and Jerry share stickers in the ratio 2:3. If Jerry has 15 stickers, how many does Tom have?", "expected_answer": "10", "topic": "ratio_proportion", "method": "finding missing ratio value"},
    {"id": "Q25", "question": "The ratio of boys to girls in a class is 4:5. If there are 20 girls, how many students are there altogether?", "expected_answer": "36", "topic": "ratio_proportion", "method": "ratio to total"},
    {"id": "Q26", "question": "Simplify the ratio 18:24.", "expected_answer": "3:4", "topic": "ratio_proportion", "method": "simplifying ratios"},
    {"id": "Q27", "question": "Ali, Ben, and Charlie share $180 in the ratio 2:3:4. How much does Ben get?", "expected_answer": "60", "topic": "ratio_proportion", "method": "three-way ratio sharing"},
    {"id": "Q28", "question": "The ratio of apples to oranges is 5:3. If there are 25 apples, how many oranges are there?", "expected_answer": "15", "topic": "ratio_proportion", "method": "equivalent ratio"},
    {"id": "Q29", "question": "Mei Ling and Raju share some money in the ratio 3:7. Raju gets $28 more than Mei Ling. How much money do they share altogether?", "expected_answer": "70", "topic": "ratio_proportion", "method": "ratio with difference"},

    # ── Rate / Unitary Reasoning (9 questions) ──
    {"id": "Q30", "question": "A car travels 180 km in 3 hours. What is its average speed?", "expected_answer": "60", "topic": "rate", "method": "speed = distance / time"},
    {"id": "Q31", "question": "5 notebooks cost $15. How much does 1 notebook cost?", "expected_answer": "3", "topic": "rate", "method": "unitary method"},
    {"id": "Q32", "question": "A printer prints 120 pages in 4 minutes. How many pages does it print per minute?", "expected_answer": "30", "topic": "rate", "method": "rate per unit"},
    {"id": "Q33", "question": "If 8 workers can build a wall in 6 days, how many days would 12 workers take?", "expected_answer": "4", "topic": "rate", "method": "inverse proportion"},
    {"id": "Q34", "question": "A tap fills a tank at 5 litres per minute. How long will it take to fill a 200-litre tank?", "expected_answer": "40", "topic": "rate", "method": "time from rate"},
    {"id": "Q35", "question": "Ali cycles at 12 km per hour. How far does he travel in 2.5 hours?", "expected_answer": "30", "topic": "rate", "method": "distance = speed x time"},
    {"id": "Q36", "question": "3 kg of rice costs $7.50. How much does 1 kg cost?", "expected_answer": "2.50", "topic": "rate", "method": "unit price"},
    {"id": "Q37", "question": "A machine produces 480 items in 8 hours. How many items does it produce in 5 hours?", "expected_answer": "300", "topic": "rate", "method": "rate then multiply"},
    {"id": "Q38", "question": "Ben types 60 words per minute. How many words can he type in 15 minutes?", "expected_answer": "900", "topic": "rate", "method": "rate multiplication"},

    # ── Measurement (9 questions) ──
    {"id": "Q39", "question": "A rectangle has length 8 cm and width 5 cm. Find its area.", "expected_answer": "40", "topic": "measurement", "method": "area of rectangle"},
    {"id": "Q40", "question": "A rectangle has length 9 cm and width 4 cm. Find its perimeter.", "expected_answer": "26", "topic": "measurement", "method": "perimeter of rectangle"},
    {"id": "Q41", "question": "Find the area of a triangle with base 10 cm and height 6 cm.", "expected_answer": "30", "topic": "measurement", "method": "area of triangle"},
    {"id": "Q42", "question": "A cube has side length 3 cm. Find its volume.", "expected_answer": "27", "topic": "measurement", "method": "volume of cube"},
    {"id": "Q43", "question": "A square has a perimeter of 36 cm. What is the length of one side?", "expected_answer": "9", "topic": "measurement", "method": "side from perimeter"},
    {"id": "Q44", "question": "Find the perimeter of a square with side 7 cm.", "expected_answer": "28", "topic": "measurement", "method": "perimeter of square"},
    {"id": "Q45", "question": "A cuboid has length 5 cm, width 3 cm, and height 4 cm. Find its volume.", "expected_answer": "60", "topic": "measurement", "method": "volume of cuboid"},
    {"id": "Q46", "question": "A rectangle has area 72 cm squared and length 9 cm. What is the width?", "expected_answer": "8", "topic": "measurement", "method": "find missing dimension"},
    {"id": "Q47", "question": "Convert 2.5 km to metres.", "expected_answer": "2500", "topic": "measurement", "method": "unit conversion"},

    # ── Data Handling (9 questions) ──
    {"id": "Q48", "question": "Find the mean of 6, 8, 10, and 12.", "expected_answer": "9", "topic": "data_handling", "method": "mean / average"},
    {"id": "Q49", "question": "A student scored 12, 15, and 18 in three tests. What is the average score?", "expected_answer": "15", "topic": "data_handling", "method": "mean / average"},
    {"id": "Q50", "question": "The average of 5 numbers is 20. What is their total sum?", "expected_answer": "100", "topic": "data_handling", "method": "sum from average"},
    {"id": "Q51", "question": "The mean of 4 numbers is 10. If three of the numbers are 8, 12, and 11, find the fourth number.", "expected_answer": "9", "topic": "data_handling", "method": "finding missing value from mean"},
    {"id": "Q52", "question": "5 students scored 70, 80, 85, 90, and 95. What is the average score?", "expected_answer": "84", "topic": "data_handling", "method": "mean / average"},
    {"id": "Q53", "question": "The total rainfall over 4 days was 60 mm. What was the average daily rainfall?", "expected_answer": "15", "topic": "data_handling", "method": "average from total"},
    {"id": "Q54", "question": "Ali's test scores are 75, 80, 90, 85, and 70. What is his average score?", "expected_answer": "80", "topic": "data_handling", "method": "mean / average"},
    {"id": "Q55", "question": "The average height of 6 students is 140 cm. If one more student of height 154 cm joins, what is the new average height?", "expected_answer": "142", "topic": "data_handling", "method": "new average after adding"},
    {"id": "Q56", "question": "A shop sold 12, 18, 15, 20, and 10 ice creams over 5 days. What was the average number sold per day?", "expected_answer": "15", "topic": "data_handling", "method": "mean / average"},
]


def extract_numeric(text: str) -> Optional[str]:
    """Extract the core numeric value from an answer string for comparison."""
    if not text:
        return None
    text = text.strip()
    frac_match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if frac_match:
        return f"{frac_match.group(1)}/{frac_match.group(2)}"
    num_match = re.search(r"-?\d+\.?\d*", text)
    if num_match:
        return num_match.group(0)
    return None


def answers_match(expected: str, actual: str) -> bool:
    """Check if the actual answer matches the expected answer."""
    exp_num = extract_numeric(expected)
    act_num = extract_numeric(actual)
    if exp_num is None or act_num is None:
        return False
    if exp_num == act_num:
        return True
    try:
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
    """Evaluate how accurately the topic classifier assigns questions to PSLE topics."""
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
        if expected not in topic_results:
            topic_results[expected] = {"correct": 0, "total": 0, "errors": []}
        topic_results[expected]["total"] += 1
        if is_correct:
            topic_results[expected]["correct"] += 1
        else:
            topic_results[expected]["errors"].append({
                "id": q["id"], "question": q["question"][:60], "predicted": predicted,
            })
        status = "PASS" if is_correct else "FAIL"
        print(f"  [{status}] {q['id']}: expected={expected}, got={predicted}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n  Overall: {correct}/{total} ({accuracy * 100:.1f}%)")
    print(f"\n  Per-topic breakdown:")
    for topic, data in sorted(topic_results.items()):
        t_acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        print(f"    {get_topic_display_name(topic)}: {data['correct']}/{data['total']} ({t_acc * 100:.0f}%)")
        for err in data["errors"]:
            print(f"      [FAIL] {err['id']}: predicted '{err['predicted']}' -- \"{err['question']}...\"")
    
    return {"accuracy": accuracy, "correct": correct, "total": total, "per_topic": topic_results}


def evaluate_retrieval_relevance(questions: List[Dict], k: int = 4) -> Dict:
    """Evaluate whether retrieved documents match the expected topic."""
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
        status = "PASS" if precision >= 0.5 else "WARN" if precision > 0 else "FAIL"
        print(f"  [{status}] {q['id']} [{q['topic']}]: {q_topic_matches}/{len(results)} on-topic, avg_sim={avg_score:.3f}")
        details.append({"id": q["id"], "topic_precision": precision, "avg_score": avg_score, "num_retrieved": len(results)})
    
    overall_precision = topic_match_docs / total_docs if total_docs > 0 else 0
    overall_avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"\n  Overall topic precision: {topic_match_docs}/{total_docs} ({overall_precision * 100:.1f}%)")
    print(f"  Overall avg similarity: {overall_avg_score:.3f}")
    
    return {"topic_precision": overall_precision, "avg_similarity": overall_avg_score, "total_docs": total_docs, "topic_match_docs": topic_match_docs, "details": details}


def evaluate_answer_correctness(questions: List[Dict], k: int = 4) -> Dict:
    """Evaluate end-to-end answer correctness using the full RAG + LLM pipeline."""
    from src.generation import answer_question, get_current_provider
    
    print("\n" + "=" * 60)
    print(f"EVALUATION 3: Answer Correctness (provider: {get_current_provider()})")
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
            status = "PASS" if is_correct else "FAIL"
            print(f"[{status}] ({elapsed:.1f}s) expected={q['expected_answer']}")
            details.append({"id": q["id"], "correct": is_correct, "expected": q["expected_answer"], "confidence": result.get("confidence", 0), "time_seconds": elapsed})
        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"[FAIL] ERROR: {str(e)[:60]}")
            details.append({"id": q["id"], "correct": False, "expected": q["expected_answer"], "error": str(e)[:100], "time_seconds": elapsed})
    
    accuracy = correct / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    print(f"\n  Accuracy: {correct}/{total} ({accuracy * 100:.1f}%)")
    print(f"  Avg time per question: {avg_time:.1f}s")
    print(f"  Total evaluation time: {total_time:.1f}s")
    
    return {"accuracy": accuracy, "correct": correct, "total": total, "avg_time_seconds": avg_time, "total_time_seconds": total_time, "details": details}


def run_evaluation(quick: bool = False, topic_filter: str = None, provider: str = None):
    """
    Run the full evaluation suite.
    
    Args:
        quick: If True, skip LLM-based answer correctness (eval 3)
        topic_filter: If set, only evaluate questions from this topic
        provider: LLM provider to use ("gemini", "openai", "groq", "ollama")
    """
    questions = BENCHMARK_QUESTIONS

    # Switch LLM provider if specified
    if provider:
        from src.generation import switch_provider
        switch_provider(provider)
        print(f"  LLM provider: {provider}")
    
    if topic_filter:
        questions = [q for q in questions if q["topic"] == topic_filter]
        if not questions:
            print(f"No benchmark questions found for topic: {topic_filter}")
            return
        print(f"Filtering to topic: {get_topic_display_name(topic_filter)} ({len(questions)} questions)")
    
    print(f"\nRunning evaluation on {len(questions)} benchmark questions...")
    
    classification_results = evaluate_topic_classification(questions)
    
    try:
        retrieval_results = evaluate_retrieval_relevance(questions)
    except Exception as e:
        print(f"\n  [FAIL] Retrieval evaluation failed: {e}")
        print("     Make sure you've built the index: python build_index.py")
        retrieval_results = None
    
    answer_results = None
    if not quick:
        try:
            answer_results = evaluate_answer_correctness(questions)
        except Exception as e:
            print(f"\n  [FAIL] Answer evaluation failed: {e}")
            print("     Make sure your API key is set in .env")
    else:
        print("\n  [SKIP] Skipping answer correctness (quick mode). Run without --quick for full eval.")
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Benchmark size: {len(questions)} questions")
    print(f"  LLM provider: {provider or 'gemini'}")
    print(f"  Topic classification accuracy: {classification_results['accuracy'] * 100:.1f}%")
    if retrieval_results:
        print(f"  Retrieval topic precision: {retrieval_results['topic_precision'] * 100:.1f}%")
        print(f"  Retrieval avg similarity: {retrieval_results['avg_similarity']:.3f}")
    if answer_results:
        print(f"  Answer correctness: {answer_results['accuracy'] * 100:.1f}%")
        print(f"  Avg response time: {answer_results['avg_time_seconds']:.1f}s")
    print("=" * 60)
    
    results = {
        "num_questions": len(questions),
        "topic_filter": topic_filter,
        "provider": provider or "gemini",
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

    csv_path = "data/benchmark/evaluation_results.csv"
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "question", "topic", "expected_answer", "method",
                "classified_topic", "classification_correct",
                "retrieval_precision", "retrieval_avg_score",
                "answer_correct", "answer_time_s", "confidence",
            ])
            for q in questions:
                classified = classify_question(q["question"])
                class_correct = classified == q["topic"]
                ret_detail = {}
                if retrieval_results:
                    ret_detail = next((d for d in retrieval_results["details"] if d["id"] == q["id"]), {})
                ans_detail = {}
                if answer_results:
                    ans_detail = next((d for d in answer_results["details"] if d["id"] == q["id"]), {})
                writer.writerow([
                    q["id"], q["question"], q["topic"], q["expected_answer"], q["method"],
                    classified, class_correct,
                    ret_detail.get("topic_precision", ""), ret_detail.get("avg_score", ""),
                    ans_detail.get("correct", ""), ans_detail.get("time_seconds", ""), ans_detail.get("confidence", ""),
                ])
        print(f"  Results CSV saved to: {csv_path}")
    except Exception as e:
        print(f"  Warning: Could not save CSV: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the PSLE Math Study Bot")
    parser.add_argument("--quick", action="store_true", help="Skip LLM-based evaluation (faster, no API calls)")
    parser.add_argument("--topic", type=str, default=None, help="Evaluate only this topic (e.g., 'percentage', 'rate')")
    parser.add_argument(
        "--provider", type=str, default=None,
        choices=["gemini", "openai", "groq", "ollama"],
        help="LLM provider for answer generation (default: gemini)",
    )
    args = parser.parse_args()
    
    run_evaluation(quick=args.quick, topic_filter=args.topic, provider=args.provider)
