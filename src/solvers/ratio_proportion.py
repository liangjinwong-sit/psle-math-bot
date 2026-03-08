from src.solvers.common import (
    extract_numbers,
    extract_ratio,
    parse_ratio_labels,
    format_number,
    format_money,
    unsupported,
)


def solve_ratio_proportion(question: str, route: dict):
    method = route.get("method")
    q = question.lower()

    if method == "share_in_ratio":
        return solve_share_in_ratio(q)

    if method == "total_from_ratio":
        return solve_total_from_ratio(q)

    if method == "equivalent_ratio":
        return solve_equivalent_ratio(q)

    return unsupported(
        "I could not determine the exact Ratio/Proportion method.",
        topic="ratio_proportion",
        method=method,
    )


def solve_share_in_ratio(question: str):
    ratio = extract_ratio(question)
    nums = extract_numbers(question)

    if not ratio or len(nums) < 1:
        return unsupported("Need a ratio and a total amount.", "ratio_proportion", "share_in_ratio")

    total = None
    for n in nums:
        if n not in ratio or nums.count(n) > ratio.count(int(n)):
            total = n
            break

    if total is None:
        total = max(nums)

    total_parts = sum(ratio)
    if total_parts == 0:
        return unsupported("Total ratio parts cannot be zero.", "ratio_proportion", "share_in_ratio")

    one_part = total / total_parts
    shares = [one_part * r for r in ratio]

    use_money = "$" in question
    final = ", ".join(format_money(x) if use_money else format_number(x) for x in shares)

    return {
        "supported": True,
        "topic": "ratio_proportion",
        "method": "share_in_ratio",
        "final": final,
        "working": (
            f"Step 1: Total ratio parts = {' + '.join(str(r) for r in ratio)} = {total_parts}.\n"
            f"Step 2: Value of 1 part = {format_number(total)} ÷ {total_parts} = {format_number(one_part)}.\n"
            f"Step 3: Multiply each ratio part by {format_number(one_part)}.\n"
            f"Answer: {final}"
        ),
        "why": "In ratio sharing, first find the value of 1 part, then multiply by each ratio number.",
    }


def solve_total_from_ratio(question: str):
    labels = parse_ratio_labels(question)
    ratio = extract_ratio(question)
    nums = extract_numbers(question)

    if labels:
        part1 = labels["part1"]
        part2 = labels["part2"]
        label1 = labels["label1"]
        label2 = labels["label2"]
    elif ratio and len(ratio) >= 2:
        part1, part2 = ratio[0], ratio[1]
        label1, label2 = "first group", "second group"
    else:
        return unsupported("Need a valid ratio.", "ratio_proportion", "total_from_ratio")

    total = None
    ratio_nums = [part1, part2]

    for n in nums:
        if n not in ratio_nums or nums.count(n) > ratio_nums.count(int(n)):
            total = n
            break

    if total is None:
        bigger = [n for n in nums if n > max(ratio_nums)]
        total = bigger[0] if bigger else None

    if total is None:
        return unsupported("Could not find the total quantity.", "ratio_proportion", "total_from_ratio")

    total_parts = part1 + part2
    if total_parts == 0:
        return unsupported("Total ratio parts cannot be zero.", "ratio_proportion", "total_from_ratio")

    one_part = total / total_parts
    amount1 = one_part * part1
    amount2 = one_part * part2

    q = question.lower()
    if "how many are girls" in q:
        chosen_label = "girls"
        chosen_value = amount2
    elif "how many are boys" in q:
        chosen_label = "boys"
        chosen_value = amount1
    elif f"how many are {label1}" in q:
        chosen_label = label1
        chosen_value = amount1
    elif f"how many are {label2}" in q:
        chosen_label = label2
        chosen_value = amount2
    else:
        chosen_label = f"{label1}: {format_number(amount1)}, {label2}: {format_number(amount2)}"
        chosen_value = None

    final = format_number(chosen_value) if chosen_value is not None else chosen_label

    return {
        "supported": True,
        "topic": "ratio_proportion",
        "method": "total_from_ratio",
        "final": final,
        "working": (
            f"Step 1: Total ratio parts = {part1} + {part2} = {total_parts}.\n"
            f"Step 2: Value of 1 part = {format_number(total)} ÷ {total_parts} = {format_number(one_part)}.\n"
            f"Step 3: {label1} = {part1} × {format_number(one_part)} = {format_number(amount1)}.\n"
            f"Step 4: {label2} = {part2} × {format_number(one_part)} = {format_number(amount2)}.\n"
            f"Answer: {final}"
        ),
        "why": "When the total and ratio are known, divide the total by the total number of parts first.",
    }


def solve_equivalent_ratio(question: str):
    ratio = extract_ratio(question)
    nums = extract_numbers(question)

    if not ratio or len(ratio) < 2:
        return unsupported("Need a valid ratio.", "ratio_proportion", "equivalent_ratio")

    a, b = ratio[0], ratio[1]

    if len(nums) < 3:
        return unsupported(
            "Need another value besides the original ratio terms.",
            "ratio_proportion",
            "equivalent_ratio",
        )

    target = None
    for n in nums:
        if n != a and n != b:
            target = n
            break

    if target is None or a == 0:
        return unsupported(
            "Could not determine the target term.",
            "ratio_proportion",
            "equivalent_ratio",
        )

    multiplier = target / a
    answer = b * multiplier

    return {
        "supported": True,
        "topic": "ratio_proportion",
        "method": "equivalent_ratio",
        "final": format_number(answer),
        "working": (
            f"Step 1: Original ratio is {a}:{b}.\n"
            f"Step 2: Multiplier = {format_number(target)} ÷ {a} = {format_number(multiplier)}.\n"
            f"Step 3: Multiply the other term by the same multiplier.\n"
            f"Step 4: {b} × {format_number(multiplier)} = {format_number(answer)}.\n"
            f"Answer: {format_number(answer)}"
        ),
        "why": "Equivalent ratios change by the same multiplier on both sides.",
    }
