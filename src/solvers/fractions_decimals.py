import re
from src.solvers.common import (
    extract_first_fraction,
    extract_all_fractions,
    decimal_to_fraction,
    fraction_to_decimal,
    format_number,
    unsupported,
)


def solve_fractions_decimals(question: str, route: dict):
    method = route.get("method")
    q = question.lower()

    if method == "fraction_to_decimal":
        return solve_fraction_to_decimal(q)

    if method == "decimal_to_fraction":
        return solve_decimal_to_fraction(q)

    if method == "compare_values":
        return solve_compare_values(q)

    if "+" in q or "-" in q:
        return solve_basic_operation(q)

    return unsupported(
        "I could not determine the exact Fractions/Decimals method.",
        topic="fractions_decimals",
        method=method,
    )


def solve_fraction_to_decimal(question: str):
    frac = extract_first_fraction(question)
    if not frac:
        return unsupported("Could not find a fraction.", "fractions_decimals", "fraction_to_decimal")

    n, d = frac
    value = fraction_to_decimal(n, d)

    return {
        "supported": True,
        "topic": "fractions_decimals",
        "method": "fraction_to_decimal",
        "final": format_number(value),
        "working": (
            f"Step 1: Fraction to decimal means numerator ÷ denominator.\n"
            f"Step 2: {n} ÷ {d} = {format_number(value)}.\n"
            f"Answer: {format_number(value)}"
        ),
        "why": "A fraction can be written as a division statement.",
    }


def solve_decimal_to_fraction(question: str):
    m = re.search(r"\d+\.\d+", question)
    if not m:
        return unsupported("Could not find a decimal.", "fractions_decimals", "decimal_to_fraction")

    value = float(m.group())
    n, d = decimal_to_fraction(value)

    return {
        "supported": True,
        "topic": "fractions_decimals",
        "method": "decimal_to_fraction",
        "final": f"{n}/{d}",
        "working": (
            f"Step 1: Write {format_number(value)} as a fraction.\n"
            f"Step 2: Simplify the fraction.\n"
            f"Step 3: {format_number(value)} = {n}/{d}.\n"
            f"Answer: {n}/{d}"
        ),
        "why": "A decimal can be written as a fraction and then simplified.",
    }


def solve_compare_values(question: str):
    fractions = extract_all_fractions(question)
    decimals = [float(x) for x in re.findall(r"\d+\.\d+", question)]

    values = []
    labels = []

    for n, d in fractions[:2]:
        values.append(n / d)
        labels.append(f"{n}/{d}")

    for d in decimals:
        values.append(d)
        labels.append(format_number(d))

    if len(values) < 2:
        return unsupported("Need two values to compare.", "fractions_decimals", "compare_values")

    a, b = values[0], values[1]
    la, lb = labels[0], labels[1]

    if "smaller" in question or "less" in question:
        final = la if a < b else lb
        relation = "smaller"
    else:
        final = la if a > b else lb
        relation = "greater"

    return {
        "supported": True,
        "topic": "fractions_decimals",
        "method": "compare_values",
        "final": final,
        "working": (
            f"Step 1: Convert both values to the same form if needed.\n"
            f"Step 2: Compare {format_number(a)} and {format_number(b)}.\n"
            f"Step 3: {final} is {relation}.\n"
            f"Answer: {final}"
        ),
        "why": "Comparing is easier when both values are written in the same form.",
    }


def solve_basic_operation(question: str):
    fractions = extract_all_fractions(question)
    decimals = [float(x) for x in re.findall(r"\d+\.\d+", question)]

    if len(fractions) >= 2:
        (n1, d1), (n2, d2) = fractions[0], fractions[1]
        op = "+" if "+" in question else "-"
        if op == "+":
            value = n1 / d1 + n2 / d2
        else:
            value = n1 / d1 - n2 / d2

        return {
            "supported": True,
            "topic": "fractions_decimals",
            "method": "operation",
            "final": format_number(value),
            "working": (
                f"Step 1: Convert the fractions into values or use a common denominator.\n"
                f"Step 2: Compute the result.\n"
                f"Step 3: Answer: {format_number(value)}"
            ),
            "why": "Fractions can be added or subtracted after rewriting them in a compatible form.",
        }

    if len(decimals) >= 2:
        a, b = decimals[0], decimals[1]
        op = "+" if "+" in question else "-"
        value = a + b if op == "+" else a - b

        return {
            "supported": True,
            "topic": "fractions_decimals",
            "method": "operation",
            "final": format_number(value),
            "working": (
                f"Step 1: Line up the decimal points.\n"
                f"Step 2: Compute {format_number(a)} {op} {format_number(b)} = {format_number(value)}.\n"
                f"Answer: {format_number(value)}"
            ),
            "why": "Decimals are added or subtracted by aligning place values.",
        }

    return unsupported("Could not parse the operation.", "fractions_decimals", "operation")
