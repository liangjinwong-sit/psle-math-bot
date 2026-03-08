# src/router.py
import re


TOPIC_METHODS = {
    "rate": [
        "unit_cost",
        "total_cost_from_unit_cost",
        "speed",
    ],
    "percentage": [
        "percentage_of_quantity",
        "percentage_increase",
        "percentage_decrease",
        "find_whole_from_percentage",
    ],
    "ratio_proportion": [
        "share_in_ratio",
        "equivalent_ratio",
        "total_from_ratio",
    ],
    "measurement": [
        "rectangle_area",
        "rectangle_perimeter",
        "square_perimeter",
        "volume",
    ],
    "data_handling": [
        "mean",
    ],
    "fractions_decimals": [
        "fraction_to_decimal",
        "decimal_to_fraction",
        "compare_values",
        "decimal_operation",
        "fraction_operation",
    ],
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("$", " $ ")
    text = text.replace("%", " % ")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_numbers(text: str):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def has_any(text: str, keywords):
    return any(k in text for k in keywords)


def count_hits(text: str, keywords):
    return sum(1 for k in keywords if k in text)


def has_fraction(text: str):
    return bool(re.search(r"\b\d+\s*/\s*\d+\b", text))


def has_decimal(text: str):
    return bool(re.search(r"\b\d+\.\d+\b", text))


def has_ratio_form(text: str):
    return bool(re.search(r"\b\d+\s*:\s*\d+\b", text))


def features(q: str):
    return {
        "numbers": extract_numbers(q),
        "has_money": "$" in q or has_any(q, ["dollar", "dollars", "cost", "costs", "price"]),
        "has_percent": "%" in q or has_any(q, ["percent", "percentage", "discount"]),
        "has_ratio": "ratio" in q or "proportion" in q or has_ratio_form(q),
        "has_fraction": has_fraction(q),
        "has_decimal": has_decimal(q) or "decimal" in q,
        "has_total": has_any(q, ["in total", "total", "altogether"]),
        "asks_how_many": has_any(q, ["how many", "how much", "what is", "find"]),
        "asks_compare": has_any(q, ["which is greater", "which is smaller", "greater", "smaller", "less", "more"]),
        "asks_convert": "convert" in q or "write" in q,
        "has_speed_units": has_any(q, ["km", "km/h", "m/s", "hour", "hours", "minute", "minutes"]),
        "has_measurement": has_any(q, ["area", "perimeter", "volume", "rectangle", "square", "length", "width", "height"]),
        "has_data": has_any(q, ["mean", "average", "table", "graph"]),
    }


def route_question(question: str):
    q = normalize(question)
    f = features(q)

    candidates = []
    candidates.extend(score_rate(q, f))
    candidates.extend(score_percentage(q, f))
    candidates.extend(score_ratio(q, f))
    candidates.extend(score_measurement(q, f))
    candidates.extend(score_data_handling(q, f))
    candidates.extend(score_fractions_decimals(q, f))

    if not candidates:
        return {
            "topic": None,
            "method": None,
            "confidence": 0.0,
            "reason": "No supported method matched.",
            "numbers": f["numbers"],
        }

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    confidence = min(best["score"] / 10.0, 0.99)

    if best["score"] < 4:
        return {
            "topic": best["topic"],
            "method": None,
            "confidence": confidence,
            "reason": f"Topic looks like {best['topic']}, but method is unclear.",
            "numbers": f["numbers"],
        }

    return {
        "topic": best["topic"],
        "method": best["method"],
        "confidence": confidence,
        "reason": best["reason"],
        "numbers": f["numbers"],
    }


def score_rate(q, f):
    results = []
    nums = f["numbers"]

    if not f["has_money"] and not f["has_speed_units"] and not has_any(q, ["each", "per", "speed"]):
        return results

    if (
        f["has_money"]
        and has_any(q, ["how much does 1", "how much does one", "how much is 1", "how much is one", "each"])
    ):
        results.append({
            "topic": "rate",
            "method": "unit_cost",
            "score": 9,
            "reason": "Detected cost question asking for cost of one item.",
        })

    if (
        f["has_money"]
        and has_any(q, ["how much do", "how much does", "how much would", "how much will", "what is the cost of"])
        and len(nums) >= 2
    ):
        results.append({
            "topic": "rate",
            "method": "total_cost_from_unit_cost",
            "score": 9,
            "reason": "Detected cost question asking for total cost from unit cost.",
        })

    if (
        f["has_money"]
        and "costs" in q
        and len(nums) >= 2
        and has_any(q, ["how much", "cost"])
    ):
        results.append({
            "topic": "rate",
            "method": "total_cost_from_unit_cost",
            "score": 7,
            "reason": "Detected item cost statement and a total-cost question.",
        })

    if (
        has_any(q, ["speed", "average speed"])
        or (f["has_speed_units"] and len(nums) >= 2)
    ):
        results.append({
            "topic": "rate",
            "method": "speed",
            "score": 8,
            "reason": "Detected distance-time rate pattern.",
        })

    return results


def score_percentage(q, f):
    results = []
    if not f["has_percent"]:
        return results

    if has_any(q, ["increase from", "percentage increase", "increases from", "increased from"]):
        results.append({
            "topic": "percentage",
            "method": "percentage_increase",
            "score": 9,
            "reason": "Detected percentage increase pattern.",
        })

    if has_any(q, ["decrease from", "percentage decrease", "decreases from", "decreased from"]):
        results.append({
            "topic": "percentage",
            "method": "percentage_decrease",
            "score": 9,
            "reason": "Detected percentage decrease pattern.",
        })

    if has_any(q, ["discount", "% of", "percent of", "find"]) and len(f["numbers"]) >= 2:
        results.append({
            "topic": "percentage",
            "method": "percentage_of_quantity",
            "score": 8,
            "reason": "Detected percentage of quantity pattern.",
        })

    if has_any(q, ["of what", "what is the whole", "find the whole"]) and len(f["numbers"]) >= 2:
        results.append({
            "topic": "percentage",
            "method": "find_whole_from_percentage",
            "score": 8,
            "reason": "Detected whole-from-percentage pattern.",
        })

    return results


def score_ratio(q, f):
    results = []
    if not f["has_ratio"]:
        return results

    if "ratio of" in q and f["has_total"] and has_any(q, ["how many are", "how many"]):
        results.append({
            "topic": "ratio_proportion",
            "method": "total_from_ratio",
            "score": 10,
            "reason": "Detected labeled ratio with total given.",
        })

    if "share" in q and (":" in q or "ratio" in q):
        results.append({
            "topic": "ratio_proportion",
            "method": "share_in_ratio",
            "score": 9,
            "reason": "Detected sharing in ratio pattern.",
        })

    if has_ratio_form(q) and has_any(q, ["equivalent", "same ratio", "first term", "second term"]):
        results.append({
            "topic": "ratio_proportion",
            "method": "equivalent_ratio",
            "score": 8,
            "reason": "Detected equivalent ratio pattern.",
        })

    if ":" in q or "ratio" in q or "proportion" in q:
        results.append({
            "topic": "ratio_proportion",
            "method": "equivalent_ratio",
            "score": 4,
            "reason": "Generic ratio/proportion match.",
        })

    return results


def score_measurement(q, f):
    results = []
    if not f["has_measurement"]:
        return results

    if "area" in q and "rectangle" in q:
        results.append({
            "topic": "measurement",
            "method": "rectangle_area",
            "score": 9,
            "reason": "Detected rectangle area pattern.",
        })

    if "perimeter" in q and "square" in q:
        results.append({
            "topic": "measurement",
            "method": "square_perimeter",
            "score": 9,
            "reason": "Detected square perimeter pattern.",
        })

    if "perimeter" in q and has_any(q, ["rectangle", "length", "width"]):
        results.append({
            "topic": "measurement",
            "method": "rectangle_perimeter",
            "score": 8,
            "reason": "Detected rectangle perimeter pattern.",
        })

    if "volume" in q or has_any(q, ["length", "width", "height"]):
        results.append({
            "topic": "measurement",
            "method": "volume",
            "score": 6,
            "reason": "Detected volume-style measurement pattern.",
        })

    return results


def score_data_handling(q, f):
    results = []
    if not f["has_data"]:
        return results

    if has_any(q, ["mean", "average"]):
        results.append({
            "topic": "data_handling",
            "method": "mean",
            "score": 9,
            "reason": "Detected mean/average pattern.",
        })

    return results


def score_fractions_decimals(q, f):
    results = []
    if not (f["has_fraction"] or f["has_decimal"] or has_any(q, ["fraction", "decimal"])):
        return results

    if f["asks_convert"] and f["has_fraction"] and "decimal" in q:
        results.append({
            "topic": "fractions_decimals",
            "method": "fraction_to_decimal",
            "score": 9,
            "reason": "Detected fraction-to-decimal conversion.",
        })

    if f["asks_convert"] and f["has_decimal"] and "fraction" in q:
        results.append({
            "topic": "fractions_decimals",
            "method": "decimal_to_fraction",
            "score": 9,
            "reason": "Detected decimal-to-fraction conversion.",
        })

    if f["asks_compare"]:
        results.append({
            "topic": "fractions_decimals",
            "method": "compare_values",
            "score": 8,
            "reason": "Detected comparison of fractions/decimals.",
        })

    if "+" in q or "-" in q:
        if f["has_fraction"]:
            results.append({
                "topic": "fractions_decimals",
                "method": "fraction_operation",
                "score": 7,
                "reason": "Detected fraction operation.",
            })
        elif f["has_decimal"]:
            results.append({
                "topic": "fractions_decimals",
                "method": "decimal_operation",
                "score": 7,
                "reason": "Detected decimal operation.",
            })

    if not results:
        results.append({
            "topic": "fractions_decimals",
            "method": "compare_values",
            "score": 4,
            "reason": "Generic fraction/decimal match.",
        })

    return results
