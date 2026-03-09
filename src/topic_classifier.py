"""
Topic classifier to categorize GSM8K questions into PSLE math topics.

PSLE Topics (6 core families):
1. Fractions & decimals (operations, comparisons, word problems)
2. Percentage (percentage of quantity, increase/decrease)
3. Ratio & proportion (comparison, sharing)
4. Rate / unitary reasoning ("per unit", multi-step)
5. Measurement (area/perimeter/volume; composite figures)
6. Data handling (tables/graphs; average/mean)
"""

import re
from typing import Dict, List


# Keywords for each PSLE topic
TOPIC_KEYWORDS = {
    "fractions_decimals": {
        "keywords": [
            "fraction", "half", "third", "quarter", "fifth", "sixth", "eighth",
            "decimal", "point", "1/2", "1/3", "1/4", "2/3", "3/4",
            "0.", "divide into", "split equally", "share equally",
        ],
        "patterns": [
            r"\d+/\d+",  # Fraction pattern like 1/2, 3/4
            r"\d+\.\d+",  # Decimal pattern like 3.5, 0.25
        ],
    },
    
    "percentage": {
        "keywords": [
            "percent", "%", "percentage", "discount", "tax", "interest",
            "increase by", "decrease by", "markup", "sale price",
            "off", "more than", "less than",
        ],
        "patterns": [
            r"\d+\s*%",  # Percentage like 20%, 5 %
            r"\d+\s*percent",  # Word form
        ],
    },
    
    "ratio_proportion": {
        "keywords": [
            "ratio", "proportion", "for every", "times as", "times more",
            "times less", "share", "distribute", "split",
            "compared to", "relationship between", "to every",
        ],
        "patterns": [
            r"\d+\s*:\s*\d+",  # Ratio pattern like 2:3, 1:4
            r"(\d+)\s+times\s+(as\s+)?(many|much|more|less|longer|shorter)",
        ],
    },
    
    "rate": {
        "keywords": [
            "per", "each", "every", "rate", "speed", "unit price",
            "miles per", "km per", "per hour", "per day", "per minute",
            "cost of one", "price of each", "how much is one",
            "per item", "per unit", "hourly", "daily", "weekly",
        ],
        "patterns": [
            r"per\s+\w+",  # per hour, per day
            r"\$\d+\s+each",  # $5 each
            r"each\s+\w+\s+(costs|is)",  # each book costs
        ],
    },
    
    "measurement": {
        "keywords": [
            "area", "perimeter", "volume", "circumference", "diameter",
            "length", "width", "height", "depth", "radius",
            "square", "rectangle", "triangle", "circle", "cube",
            "meters", "feet", "inches", "cm", "m²", "cm²",
            "square meters", "square feet", "cubic",
        ],
        "patterns": [
            r"\d+\s*(meter|metre|foot|feet|inch|cm|mm|km)",
            r"(square|cubic)\s+\w+",
        ],
    },
    
    "data_handling": {
        "keywords": [
            "average", "mean", "median", "mode", "total",
            "sum", "chart", "graph", "table", "data",
            "score", "grade", "test", "exam",
            "altogether", "combined", "in total",
        ],
        "patterns": [
            r"average\s+of",
            r"mean\s+of",
        ],
    },
}


def classify_question(question: str) -> str:
    """
    Classify a GSM8K question into one of the 6 PSLE topics.
    
    Args:
        question: The math question text
    
    Returns:
        Topic name (e.g., "percentage", "fractions_decimals", etc.)
        Returns "general" if no clear match
    """
    question_lower = question.lower()
    
    # Score each topic
    topic_scores = {}
    
    for topic, data in TOPIC_KEYWORDS.items():
        score = 0
        
        # Check keywords
        for keyword in data["keywords"]:
            if keyword in question_lower:
                score += 1
        
        # Check regex patterns
        for pattern in data.get("patterns", []):
            if re.search(pattern, question_lower):
                score += 2  # Patterns are stronger indicators
        
        topic_scores[topic] = score
    
    # Get topic with highest score
    if max(topic_scores.values()) > 0:
        best_topic = max(topic_scores, key=topic_scores.get)
        return best_topic
    
    # Default to general if no match
    return "general"


def get_topic_display_name(topic_key: str) -> str:
    """Convert topic key to display name."""
    topic_names = {
        "fractions_decimals": "Fractions & Decimals",
        "percentage": "Percentage",
        "ratio_proportion": "Ratio & Proportion",
        "rate": "Rate / Unitary Reasoning",
        "measurement": "Measurement",
        "data_handling": "Data Handling",
        "general": "General",
    }
    return topic_names.get(topic_key, topic_key)


def get_all_topics() -> List[Dict[str, str]]:
    """Get list of all PSLE topics for UI display."""
    return [
        {"key": "fractions_decimals", "name": "Fractions & Decimals", "description": "Operations, comparisons, word problems"},
        {"key": "percentage", "name": "Percentage", "description": "Percentage of quantity, increase/decrease"},
        {"key": "ratio_proportion", "name": "Ratio & Proportion", "description": "Comparison, sharing"},
        {"key": "rate", "name": "Rate / Unitary Reasoning", "description": "Per unit, multi-step"},
        {"key": "measurement", "name": "Measurement", "description": "Area, perimeter, volume"},
        {"key": "data_handling", "name": "Data Handling", "description": "Tables, graphs, average, mean"},
    ]


if __name__ == "__main__":
    # Test the classifier
    test_questions = [
        ("Janet's ducks lay 16 eggs per day. She eats three for breakfast.", "rate"),
        ("A shirt costs $60 and is sold at a 20% discount.", "percentage"),
        ("What is 1/4 + 2/4?", "fractions_decimals"),
        ("The ratio of boys to girls is 3:2.", "ratio_proportion"),
        ("Find the area of a rectangle with length 5m and width 3m.", "measurement"),
        ("The average score of 5 students is 80.", "data_handling"),
    ]
    
    print("Testing Topic Classifier:")
    print("="*60)
    for question, expected in test_questions:
        classified = classify_question(question)
        status = "✅" if classified == expected else "❌"
        print(f"{status} '{question[:50]}...'")
        print(f"   Expected: {expected}, Got: {classified}")
        print()
