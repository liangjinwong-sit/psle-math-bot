from src.solvers.common import (
    extract_numbers,
    extract_percentage,
    format_number,
    format_money,
    unsupported,
)


def solve_percentage(question: str, route: dict):
    method = route.get("method")
    q = question.lower()

    if method == "percentage_of_quantity":
        return solve_percentage_of_quantity(q)

    if method == "percentage_increase":
        return solve_percentage_change(q, increase=True)

    if method == "percentage_decrease":
        return solve_percentage_change(q, increase=False)

    if method == "find_whole_from_percentage":
        return solve_find_whole(q)

    return unsupported(
        "I could not determine the exact Percentage method.",
        topic="percentage",
        method=method,
    )


def solve_percentage_of_quantity(question: str):
    percent = extract_percentage(question)
    nums = extract_numbers(question)

    if percent is None:
        return unsupported(
            "Could not find the percentage value.",
            "percentage",
            "percentage_of_quantity",
        )

    candidates = [x for x in nums if x != percent]
    if not candidates:
        return unsupported(
            "Could not find the quantity for the percentage calculation.",
            "percentage",
            "percentage_of_quantity",
        )

    quantity = candidates[0]
    answer = percent / 100 * quantity

    is_money = "$" in question or "cost" in question or "discount" in question
    final = format_money(answer) if is_money else format_number(answer)

    return {
        "supported": True,
        "topic": "percentage",
        "method": "percentage_of_quantity",
        "final": final,
        "working": (
            f"Step 1: Convert {format_number(percent)}% to decimal: {format_number(percent / 100)}.\n"
            f"Step 2: Multiply the quantity by the decimal.\n"
            f"Step 3: {format_number(quantity)} × {format_number(percent / 100)} = {format_number(answer)}.\n"
            f"Answer: {final}"
        ),
        "why": "To find a percentage of a quantity, multiply the whole amount by the percentage in decimal form.",
    }


def solve_percentage_change(question: str, increase=True):
    nums = extract_numbers(question)
    if len(nums) < 2:
        return unsupported(
            "Need the original and new amounts.",
            "percentage",
            "percentage_increase" if increase else "percentage_decrease",
        )

    original = nums[0]
    new = nums[1]

    if original == 0:
        return unsupported(
            "Cannot divide by zero original amount.",
            "percentage",
            "percentage_increase" if increase else "percentage_decrease",
        )

    change = new - original if increase else original - new
    pct = change / original * 100

    method = "percentage_increase" if increase else "percentage_decrease"

    return {
        "supported": True,
        "topic": "percentage",
        "method": method,
        "final": f"{format_number(pct)}%",
        "working": (
            f"Step 1: Find the change.\n"
            f"Step 2: {format_number(change)}.\n"
            f"Step 3: Percentage change = change ÷ original × 100.\n"
            f"Step 4: {format_number(change)} ÷ {format_number(original)} × 100 = {format_number(pct)}%.\n"
            f"Answer: {format_number(pct)}%"
        ),
        "why": "Percentage change is always compared with the original amount.",
    }


def solve_find_whole(question: str):
    percent = extract_percentage(question)
    nums = extract_numbers(question)

    if percent is None or not nums:
        return unsupported(
            "Could not parse the percentage and part.",
            "percentage",
            "find_whole_from_percentage",
        )

    part = None
    for n in nums:
        if n != percent:
            part = n
            break

    if part is None or percent == 0:
        return unsupported(
            "Could not determine the whole from the given values.",
            "percentage",
            "find_whole_from_percentage",
        )

    whole = part / (percent / 100)

    return {
        "supported": True,
        "topic": "percentage",
        "method": "find_whole_from_percentage",
        "final": format_number(whole),
        "working": (
            f"Step 1: Convert {format_number(percent)}% to decimal: {format_number(percent / 100)}.\n"
            f"Step 2: Whole = part ÷ decimal percentage.\n"
            f"Step 3: {format_number(part)} ÷ {format_number(percent / 100)} = {format_number(whole)}.\n"
            f"Answer: {format_number(whole)}"
        ),
        "why": "If a part and its percentage are known, divide the part by the decimal percentage to find the whole.",
    }
