from src.solvers.common import extract_numbers, format_number, unsupported


def solve_data_handling(question: str, route: dict):
    method = route.get("method")
    nums = route.get("numbers") or extract_numbers(question)

    if method == "mean":
        return solve_mean(nums)

    return unsupported(
        "I could not determine the exact Data Handling method.",
        topic="data_handling",
        method=method,
    )


def solve_mean(nums):
    if not nums:
        return unsupported("Need at least one number to find the mean.", "data_handling", "mean")

    total = sum(nums)
    count = len(nums)
    mean = total / count

    return {
        "supported": True,
        "topic": "data_handling",
        "method": "mean",
        "final": format_number(mean),
        "working": (
            f"Step 1: Add all the values: {format_number(total)}.\n"
            f"Step 2: Count the number of values: {count}.\n"
            f"Step 3: Mean = total ÷ number of values.\n"
            f"Step 4: {format_number(total)} ÷ {count} = {format_number(mean)}.\n"
            f"Answer: {format_number(mean)}"
        ),
        "why": "The mean is the total of all values divided by how many values there are.",
    }
