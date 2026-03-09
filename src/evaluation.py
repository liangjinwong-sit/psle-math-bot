"""
Evaluation module for assessing RAG pipeline performance on benchmark questions.

This module helps evaluate:
1. Retrieval accuracy (are the right documents retrieved?)
2. Answer correctness (does the LLM produce the expected answer?)
3. Response quality (is the explanation clear and complete?)
"""

import re
from src.generation import answer_question
from src.retrieval import get_retriever


def extract_answer_from_response(response_text: str):
    """
    Extract the final numeric answer from the LLM response.
    Looks for patterns like "Final Answer: 20" or just numbers.
    """
    # Try to find "Final Answer:" section
    final_answer_match = re.search(
        r'(?:Final Answer|Answer):\s*\*?\*?([^\n]+)', 
        response_text, 
        re.IGNORECASE
    )
    if final_answer_match:
        answer = final_answer_match.group(1).strip()
        # Clean markdown formatting
        answer = answer.replace("**", "").strip()
        return answer
    
    # Fallback: look for the first number or currency value
    number_match = re.search(r'[\$]?\d+(?:\.\d+)?%?', response_text)
    if number_match:
        return number_match.group(0)
    
    return response_text[:100]  # Return first 100 chars if can't parse


def check_retrieval_quality(question: str, expected_topic: str, k: int = 4):
    """
    Check if the retrieved documents match the expected topic.
    
    Returns:
        dict with retrieval metrics
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(question)
    
    topics_retrieved = [doc.metadata.get("topic", "") for doc in docs]
    sources_retrieved = [doc.metadata.get("source", "") for doc in docs]
    
    # Check if expected topic appears in retrieved docs
    topic_match_count = sum(1 for t in topics_retrieved if expected_topic.lower() in t.lower())
    topic_match_rate = topic_match_count / len(docs) if docs else 0
    
    return {
        "num_docs": len(docs),
        "topics_retrieved": topics_retrieved,
        "sources_retrieved": sources_retrieved,
        "expected_topic": expected_topic,
        "topic_match_count": topic_match_count,
        "topic_match_rate": topic_match_rate,
        "passed": topic_match_rate >= 0.5,  # At least 50% should match
    }


def evaluate_single_question(
    question: str,
    expected_answer: str,
    expected_topic: str,
    verbose: bool = False,
):
    """
    Evaluate a single benchmark question.
    
    Returns:
        dict with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Expected Topic: {expected_topic}")
    print(f"{'='*60}")
    
    # Step 1: Check retrieval quality
    retrieval_result = check_retrieval_quality(question, expected_topic, k=4)
    
    if verbose:
        print(f"\n📚 Retrieval Quality:")
        print(f"  - Documents retrieved: {retrieval_result['num_docs']}")
        print(f"  - Topics: {', '.join(retrieval_result['topics_retrieved'])}")
        print(f"  - Topic match rate: {retrieval_result['topic_match_rate']:.1%}")
        print(f"  - Passed: {'✅' if retrieval_result['passed'] else '❌'}")
    
    # Step 2: Generate answer
    response = answer_question(question, k=4)
    generated_answer_full = response["answer"]
    generated_answer_extracted = extract_answer_from_response(generated_answer_full)
    
    # Step 3: Check answer correctness (simple string matching)
    # Normalize both answers for comparison
    expected_norm = expected_answer.lower().replace(" ", "").replace("$", "")
    generated_norm = generated_answer_extracted.lower().replace(" ", "").replace("$", "")
    
    answer_correct = expected_norm in generated_norm or generated_norm in expected_norm
    
    if verbose:
        print(f"\n💡 Generated Answer:")
        print(f"  - Extracted: {generated_answer_extracted}")
        print(f"  - Correct: {'✅' if answer_correct else '❌'}")
        print(f"\n📝 Full Response:")
        print(generated_answer_full[:300] + "..." if len(generated_answer_full) > 300 else generated_answer_full)
    
    return {
        "question": question,
        "expected_answer": expected_answer,
        "expected_topic": expected_topic,
        "generated_answer_extracted": generated_answer_extracted,
        "generated_answer_full": generated_answer_full,
        "retrieval_quality": retrieval_result,
        "answer_correct": answer_correct,
        "overall_passed": retrieval_result["passed"] and answer_correct,
    }


def run_benchmark_evaluation(benchmark_questions: list, verbose: bool = True):
    """
    Run evaluation on a list of benchmark questions.
    
    Args:
        benchmark_questions: List of dicts with keys: question, expected_answer, expected_topic
        verbose: Print detailed results for each question
    
    Returns:
        Summary dict with overall metrics
    """
    results = []
    
    for i, bq in enumerate(benchmark_questions, 1):
        print(f"\n\n{'#'*60}")
        print(f"Evaluating Question {i}/{len(benchmark_questions)}")
        print(f"{'#'*60}")
        
        result = evaluate_single_question(
            question=bq["question"],
            expected_answer=bq["expected_answer"],
            expected_topic=bq["expected_topic"],
            verbose=verbose,
        )
        results.append(result)
    
    # Calculate summary statistics
    total = len(results)
    retrieval_passed = sum(1 for r in results if r["retrieval_quality"]["passed"])
    answer_correct = sum(1 for r in results if r["answer_correct"])
    overall_passed = sum(1 for r in results if r["overall_passed"])
    
    avg_topic_match = sum(
        r["retrieval_quality"]["topic_match_rate"] for r in results
    ) / total if total > 0 else 0
    
    summary = {
        "total_questions": total,
        "retrieval_passed": retrieval_passed,
        "retrieval_pass_rate": retrieval_passed / total if total > 0 else 0,
        "answers_correct": answer_correct,
        "answer_accuracy": answer_correct / total if total > 0 else 0,
        "overall_passed": overall_passed,
        "overall_pass_rate": overall_passed / total if total > 0 else 0,
        "avg_topic_match_rate": avg_topic_match,
        "detailed_results": results,
    }
    
    # Print summary
    print(f"\n\n{'='*60}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Questions: {total}")
    print(f"\n🔍 Retrieval Performance:")
    print(f"  - Passed: {retrieval_passed}/{total} ({summary['retrieval_pass_rate']:.1%})")
    print(f"  - Avg Topic Match Rate: {avg_topic_match:.1%}")
    print(f"\n✅ Answer Accuracy:")
    print(f"  - Correct: {answer_correct}/{total} ({summary['answer_accuracy']:.1%})")
    print(f"\n🎯 Overall Performance:")
    print(f"  - Fully Passed: {overall_passed}/{total} ({summary['overall_pass_rate']:.1%})")
    print(f"{'='*60}\n")
    
    return summary


if __name__ == "__main__":
    # Example benchmark questions from your benchmark_questions.md
    sample_benchmark = [
        {
            "question": "Find 25% of 80.",
            "expected_answer": "20",
            "expected_topic": "percentage",
        },
        {
            "question": "A shirt costs $60 and is sold at a 20% discount. How much is the discount?",
            "expected_answer": "$12",
            "expected_topic": "percentage",
        },
        {
            "question": "Calculate 1/4 + 2/4.",
            "expected_answer": "3/4",
            "expected_topic": "fractions",
        },
    ]
    
    # Run evaluation
    results = run_benchmark_evaluation(sample_benchmark, verbose=True)
