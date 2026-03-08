import re
from fractions import Fraction


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_numbers(text: str):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def extract_percentage(text: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1))

    m = re.search(r"(\d+(?:\.\d+)?)\s*percent", text.lower())
    if m:
        return float(m.group(1))

    return None


def extract_first_fraction(text: str):
    m = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def extract_all_fractions(text: str):
    matches = re.findall(r"(\d+)\s*/\s*(\d+)", text)
    return [(int(a), int(b)) for a, b in matches]


def extract_ratio(text: str):
    m = re.search(r"(\d+)\s*:\s*(\d+)(?:\s*:\s*(\d+))?", text)
    if not m:
        return None
    return [int(x) for x in m.groups() if x is not None]


def parse_ratio_labels(text: str):
    m = re.search(
        r"ratio of\s+([a-z ]+?)\s+to\s+([a-z ]+?)\s+is\s+(\d+)\s*:\s*(\d+)",
        text.lower(),
    )
    if not m:
        return None

    return {
        "label1": m.group(1).strip(),
        "label2": m.group(2).strip(),
        "part1": int(m.group(3)),
        "part2": int(m.group(4)),
    }


def format_number(x):
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def format_money(x):
    return f"${float(x):.2f}".rstrip("0").rstrip(".")


def fraction_to_decimal(n, d):
    if d == 0:
        raise ZeroDivisionError("Denominator cannot be zero.")
    return n / d


def decimal_to_fraction(x):
    frac = Fraction(str(x)).limit_denominator()
    return frac.numerator, frac.denominator


def unsupported(message, topic=None, method=None):
    return {
        "supported": False,
        "topic": topic,
        "method": method,
        "final": None,
        "working": message,
        "why": None,
    }
