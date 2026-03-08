from src.solvers.common import extract_numbers, format_money, format_number, unsupported


def solve_rate(question: str, route: dict):
    method = route.get("method")
    nums = route.get("numbers") or extract_numbers(question)

    if method == "unit_cost":
        return solve_unit_cost(nums)

    if method == "total_cost_from_unit_cost":
        return solve_total_cost(nums)

    if method == "speed":
        return solve_speed(nums)

    return unsupported(
        "I could not determine the exact Rate method.",
        topic="rate",
        method=method,
    )


def solve_unit_cost(nums):
    if len(nums) < 2:
        return unsupported("Need at least 2 numbers for unit cost.", "rate", "unit_cost")

    number_of_items = nums[0]
    total_cost = nums[1]

    if number_of_items == 0:
        return unsupported("Cannot divide by zero items.", "rate", "unit_cost")

    unit_cost = total_cost / number_of_items

    return {
        "supported": True,
        "topic": "rate",
        "method": "unit_cost",
        "final": format_money(unit_cost),
        "working": (
            f"Step 1: {format_number(number_of_items)} items cost {format_money(total_cost)}.\n"
            f"Step 2: Cost of 1 item = total cost ÷ number of items.\n"
            f"Step 3: {format_money(total_cost)} ÷ {format_number(number_of_items)} = {format_number(unit_cost)}.\n"
            f"Answer: {format_money(unit_cost)}"
        ),
        "why": "To find the cost of 1 item, divide the total cost by the number of items.",
    }


def solve_total_cost(nums):
    if len(nums) < 2:
        return unsupported(
            "Need at least 2 numbers for total cost.",
            "rate",
            "total_cost_from_unit_cost",
        )

    unit_cost = nums[0]
    quantity = nums[1]
    total_cost = unit_cost * quantity

    return {
        "supported": True,
        "topic": "rate",
        "method": "total_cost_from_unit_cost",
        "final": format_money(total_cost),
        "working": (
            f"Step 1: Cost of 1 item = {format_money(unit_cost)}.\n"
            f"Step 2: Number of items = {format_number(quantity)}.\n"
            f"Step 3: Total cost = cost of 1 item × number of items.\n"
            f"Step 4: {format_number(unit_cost)} × {format_number(quantity)} = {format_number(total_cost)}.\n"
            f"Answer: {format_money(total_cost)}"
        ),
        "why": "If each item costs the same amount, multiply the unit cost by the number of items.",
    }


def solve_speed(nums):
    if len(nums) < 2:
        return unsupported("Need distance and time for speed.", "rate", "speed")

    distance = nums[0]
    time = nums[1]

    if time == 0:
        return unsupported("Cannot divide by zero time.", "rate", "speed")

    speed = distance / time

    return {
        "supported": True,
        "topic": "rate",
        "method": "speed",
        "final": format_number(speed),
        "working": (
            f"Step 1: Speed = distance ÷ time.\n"
            f"Step 2: {format_number(distance)} ÷ {format_number(time)} = {format_number(speed)}.\n"
            f"Answer: {format_number(speed)}"
        ),
        "why": "A rate tells us how much per 1 unit, so speed is distance per unit time.",
    }
