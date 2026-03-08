#!/usr/bin/env python3
"""
Test script to verify the local solver integration works without any external APIs.
"""

from src.router import route_question
from src.solvers import solve_question_by_route


def test_question(question: str):
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}")
    
    # Route the question
    route = route_question(question)
    print(f"\n📍 ROUTING:")
    print(f"   Topic: {route.get('topic')}")
    print(f"   Method: {route.get('method')}")
    print(f"   Confidence: {route.get('confidence', 0):.2%}")
    print(f"   Reason: {route.get('reason')}")
    
    # Solve the question
    if route.get('topic'):
        result = solve_question_by_route(question, route)
        
        print(f"\n✅ SOLUTION:")
        print(f"   Supported: {result.get('supported')}")
        
        if result.get('final'):
            print(f"\n📊 FINAL ANSWER:")
            print(f"   {result.get('final')}")
        
        if result.get('working'):
            print(f"\n🔢 WORKING:")
            for line in result.get('working').split('\n'):
                print(f"   {line}")
        
        if result.get('why'):
            print(f"\n💡 EXPLANATION:")
            print(f"   {result.get('why')}")
    else:
        print("\n❌ No solver found for this question type.")


def main():
    print("\n" + "="*80)
    print("PSLE MATH BOT - LOCAL SOLVER TEST")
    print("Testing without any external LLM APIs")
    print("="*80)
    
    test_questions = [
        # Percentage
        "Find 20% of 150.",
        
        # Rate
        "5 notebooks cost $15. How much does 1 notebook cost?",
        
        # Ratio
        "Share $120 in the ratio 2:3.",
        
        # Measurement
        "What is the area of a rectangle with length 8 cm and width 5 cm?",
        
        # Fractions
        "Convert 3/4 to a decimal.",
    ]
    
    for question in test_questions:
        test_question(question)
    
    print(f"\n{'='*80}")
    print("✅ All tests completed! The bot is running fully locally.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
