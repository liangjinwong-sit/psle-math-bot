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


# Keywords for each PSLE topic, with weights
# Higher weight = stronger indicator for that topic
TOPIC_KEYWORDS = {
    "fractions_decimals": {
        "keywords": [
            "fraction", "half", "third", "quarter", "fifth", "sixth", "eighth",
            "decimal", "1/2", "1/3", "1/4", "2/3", "3/4",
            "divide into", "split equally", "share equally",
            "simplest form", "equivalent fraction", "mixed number",
            "numerator", "denominator", "improper fraction",
            "which is smaller", "which is greater", "which is larger",
        ],
        "patterns": [
            r"\b\d+\s*/\s*\d+\b",           # Fraction pattern like 1/2, 3/4
            r"(?<!\$)\b\d+\.\d+\b",          # Decimal not preceded by $ (avoids prices)
        ],
    },
    
    "percentage": {
        "keywords": [
            "percent", "percentage", "discount", "tax", "interest",
            "increase by", "decrease by", "markup", "sale price",
            "gst", "profit margin", "as a percentage",
        ],
        "patterns": [
            r"\d+\s*%",           # Percentage like 20%, 5%
            r"\d+\s*percent",     # Word form
        ],
    },
    
    "ratio_proportion": {
        "keywords": [
            "ratio", "proportion", "for every", "times as many",
            "times as much", "times more", "times less",
            "share in the ratio", "distribute", "divided in the ratio",
            "compared to", "to every", "simplify the ratio",
        ],
        "patterns": [
            r"\b\d+\s*:\s*\d+\b",  # Ratio pattern like 2:3, 1:4
            r"(\d+)\s+times\s+(as\s+)?(many|much|more|less|longer|shorter)",
        ],
    },
    
    "rate": {
        "keywords": [
            "per hour", "per day", "per minute", "per week", "per month",
            "per litre", "per kilogram", "per item", "per unit",
            "rate", "speed", "average speed", "unit price", "unit cost",
            "miles per", "km per", "kilometers per",
            "cost of one", "price of each", "how much is one",
            "how much does 1", "how much does one",
            "hourly", "daily", "weekly", "words per minute",
            "workers", "machines", "taps",
        ],
        "patterns": [
            r"\bper\s+(hour|day|minute|second|week|month|year|litre|liter|kg|kilogram|item|unit|person)\b",
            r"\$\d+\.?\d*\s+each\b",                          # $5 each
            r"\beach\s+\w+\s+(costs?|is)\b",                   # each book costs
            r"\bin\s+\d+\s+(hours?|days?|minutes?|seconds?|weeks?|months?|years?)\b",  # in 3 hours
            r"\d+\s+\w+\s+costs?\s+\$",                       # 5 notebooks cost $
            r"\d+\s+kg\b.*costs?\b",                           # 3 kg ... costs
            r"\bproduces?\s+\d+",                              # produces 480
            r"\btypes?\s+\d+\s+words",                         # types 60 words
        ],
    },
    
    "measurement": {
        "keywords": [
            "area", "perimeter", "volume", "circumference", "diameter",
            "length", "width", "height", "depth", "radius",
            "rectangle", "triangle", "circle", "cube", "cuboid",
            "square meters", "square feet", "cubic",
            "composite figure", "shaded region",
            "convert", "metres", "meters", "centimetres",
        ],
        "patterns": [
            r"\b\d+\s*(cm|mm|km|m)\b",  # Measurements (but not inside other words)
            r"\b(square|cubic)\s+\w+",
            r"\bm[²³]\b|\bcm[²³]\b",  # m², cm³ etc.
            r"cm\s+squared",           # "cm squared"
        ],
    },
    
    "data_handling": {
        "keywords": [
            "average", "mean", "median", "mode",
            "bar chart", "pie chart", "line graph", "bar graph",
            "table shows", "data", "tally",
            "total number of", "how many more",
            "average score", "average height", "average daily",
            "average number",
        ],
        "patterns": [
            r"\baverage\b",
            r"\bmean\b",
            r"\bscored?\s+\d+.*\d+.*\d+",  # scored X, Y, Z (list of scores)
        ],
    },
}


def classify_question(question: str) -> str:
    """
    Classify a math question into one of the 6 PSLE topics.
    
    Uses keyword matching and regex patterns to score each topic.
    Patterns are weighted higher (3 points) than keywords (1 point)
    to reduce false positives from common words.
    
    Args:
        question: The math question text
    
    Returns:
        Topic key string (e.g., "percentage", "fractions_decimals")
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
        
        # Check regex patterns (stronger indicators)
        for pattern in data.get("patterns", []):
            if re.search(pattern, question_lower):
                score += 3
        
        topic_scores[topic] = score
    
    # Get topic with highest score
    max_score = max(topic_scores.values())
    if max_score > 0:
        best_topic = max(topic_scores, key=topic_scores.get)
        return best_topic
    
    # Default to general if no match
    return "general"


def get_topic_display_name(topic_key: str) -> str:
    """Convert topic key to user-friendly display name."""
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
    # Test the classifier with edge cases
    test_questions = [
        ("Janet's ducks lay 16 eggs per day. She eats three for breakfast.", "rate"),
        ("A shirt costs $60 and is sold at a 20% discount.", "percentage"),
        ("What is 1/4 + 2/4?", "fractions_decimals"),
        ("The ratio of boys to girls is 3:2.", "ratio_proportion"),
        ("Find the area of a rectangle with length 5m and width 3m.", "measurement"),
        ("The average score of 5 students is 80.", "data_handling"),
        ("A toy costs $5.50. If you buy 3, how much do you pay?", "general"),
        ("What is 25 percent of 80?", "percentage"),
        ("John scored 85% on his test.", "percentage"),
        # New benchmark edge cases
        ("Calculate 4.25 - 1.8.", "fractions_decimals"),
        ("Which is smaller, 0.48 or 0.5?", "fractions_decimals"),
        ("A car travels 180 km in 3 hours. What is its average speed?", "rate"),
        ("5 notebooks cost $15. How much does 1 notebook cost?", "rate"),
        ("If 8 workers can build a wall in 6 days, how many days would 12 workers take?", "rate"),
        ("3 kg of rice costs $7.50. How much does 1 kg cost?", "rate"),
        ("A machine produces 480 items in 8 hours.", "rate"),
        ("The total rainfall over 4 days was 60 mm. What was the average daily rainfall?", "data_handling"),
        ("The average height of 6 students is 140 cm.", "data_handling"),
        ("A shop sold 12, 18, 15, 20, and 10 ice creams over 5 days. What was the average number sold per day?", "data_handling"),
    ]
    
    print("Testing Topic Classifier:")
    print("=" * 60)
    correct = 0
    for question, expected in test_questions:
        classified = classify_question(question)
        status = "PASS" if classified == expected else "FAIL"
        if classified == expected:
            correct += 1
        print(f"[{status}] '{question[:55]}...'")
        print(f"   Expected: {expected}, Got: {classified}")
        print()
    print(f"Accuracy: {correct}/{len(test_questions)} ({correct/len(test_questions)*100:.0f}%)")
