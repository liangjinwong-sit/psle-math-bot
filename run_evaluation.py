"""
Parse benchmark questions from markdown and run evaluation.

This script reads your data/benchmark/benchmark_questions.md file,
extracts all questions, and runs automated evaluation.
"""

import re
from src.evaluation import run_benchmark_evaluation


def parse_benchmark_markdown(filepath="data/benchmark/benchmark_questions.md"):
    """
    Parse benchmark questions from markdown file.
    
    Expected format:
    ## Q1
    Question: ...
    Topic: ...
    Expected answer: ...
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by ## Q markers
    question_blocks = re.split(r'##\s+Q\d+', content)[1:]  # Skip header
    
    benchmarks = []
    for block in question_blocks:
        # Extract fields
        question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', block)
        topic_match = re.search(r'Topic:\s*(.+?)(?:\n|$)', block)
        answer_match = re.search(r'Expected answer:\s*(.+?)(?:\n|$)', block)
        
        if question_match and topic_match and answer_match:
            benchmarks.append({
                "question": question_match.group(1).strip(),
                "expected_topic": topic_match.group(1).strip().lower(),
                "expected_answer": answer_match.group(1).strip(),
            })
    
    print(f"✅ Parsed {len(benchmarks)} benchmark questions from {filepath}")
    return benchmarks


if __name__ == "__main__":
    print("="*60)
    print("PSLE Math Bot - Benchmark Evaluation")
    print("="*60)
    
    # Parse benchmark questions
    benchmarks = parse_benchmark_markdown()
    
    if not benchmarks:
        print("❌ No benchmark questions found.")
        print("   Please check data/benchmark/benchmark_questions.md exists and is properly formatted.")
        exit(1)
    
    print(f"\n📋 Found {len(benchmarks)} questions to evaluate:")
    for i, bq in enumerate(benchmarks[:3], 1):
        print(f"  {i}. {bq['question'][:60]}... (Topic: {bq['expected_topic']})")
    if len(benchmarks) > 3:
        print(f"  ... and {len(benchmarks) - 3} more")
    
    # Confirm before running
    print("\n⚠️  This will make LLM API calls and may take several minutes.")
    response = input("Continue? (y/n): ")
    
    if response.lower() != 'y':
        print("Cancelled.")
        exit(0)
    
    # Run evaluation
    results = run_benchmark_evaluation(benchmarks, verbose=True)
    
    # Save results to file
    output_file = "evaluation_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PSLE Math Bot - Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Questions: {results['total_questions']}\n\n")
        
        f.write("🔍 Retrieval Performance:\n")
        f.write(f"  - Passed: {results['retrieval_passed']}/{results['total_questions']} ")
        f.write(f"({results['retrieval_pass_rate']:.1%})\n")
        f.write(f"  - Avg Topic Match: {results['avg_topic_match_rate']:.1%}\n\n")
        
        f.write("✅ Answer Accuracy:\n")
        f.write(f"  - Correct: {results['answers_correct']}/{results['total_questions']} ")
        f.write(f"({results['answer_accuracy']:.1%})\n\n")
        
        f.write("🎯 Overall Performance:\n")
        f.write(f"  - Fully Passed: {results['overall_passed']}/{results['total_questions']} ")
        f.write(f"({results['overall_pass_rate']:.1%})\n\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Detailed Results\n")
        f.write("="*60 + "\n\n")
        
        for i, r in enumerate(results['detailed_results'], 1):
            f.write(f"\nQuestion {i}:\n")
            f.write(f"  Q: {r['question']}\n")
            f.write(f"  Expected: {r['expected_answer']}\n")
            f.write(f"  Generated: {r['generated_answer_extracted']}\n")
            f.write(f"  Correct: {'✅' if r['answer_correct'] else '❌'}\n")
            f.write(f"  Retrieval: {'✅' if r['retrieval_quality']['passed'] else '❌'}\n")
            f.write(f"  Overall: {'✅ PASS' if r['overall_passed'] else '❌ FAIL'}\n")
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n🎉 Evaluation complete!")
