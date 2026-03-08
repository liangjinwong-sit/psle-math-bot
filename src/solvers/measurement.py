from src.solvers.common import extract_numbers, format_number, unsupported


def solve_measurement(question: str, route: dict):
    method = route.get("method")
    nums = route.get("numbers") or extract_numbers(question)

    if method == "rectangle_area":
        return solve_rectangle_area(nums)

    if method == "rectangle_perimeter":
        return solve_rectangle_perimeter(nums)

    if method == "square_perimeter":
        return solve_square_perimeter(nums)

    if method == "volume":
        return solve_volume(nums)

    return unsupported(
        "I could not determine the exact Measurement method.",
        topic="measurement",
        method=method,
    )


def solve_rectangle_area(nums):
    if len(nums) < 2:
        return unsupported("Need length and width.", "measurement", "rectangle_area")

    length, width = nums[0], nums[1]
    area = length * width

    return {
        "supported": True,
        "topic": "measurement",
        "method": "rectangle_area",
        "final": format_number(area),
        "working": (
            f"Step 1: Area of rectangle = length × width.\n"
            f"Step 2: {format_number(length)} × {format_number(width)} = {format_number(area)}.\n"
            f"Answer: {format_number(area)}"
        ),
        "why": "A rectangle’s area is found by multiplying its length by its width.",
    }


def solve_rectangle_perimeter(nums):
    if len(nums) < 2:
        return unsupported("Need length and width.", "measurement", "rectangle_perimeter")

    length, width = nums[0], nums[1]
    p = 2 * (length + width)

    return {
        "supported": True,
        "topic": "measurement",
        "method": "rectangle_perimeter",
        "final": format_number(p),
        "working": (
            f"Step 1: Perimeter of rectangle = 2 × (length + width).\n"
            f"Step 2: 2 × ({format_number(length)} + {format_number(width)}) = {format_number(p)}.\n"
            f"Answer: {format_number(p)}"
        ),
        "why": "Perimeter is the total distance around the shape.",
    }


def solve_square_perimeter(nums):
    if len(nums) < 1:
        return unsupported("Need the side length.", "measurement", "square_perimeter")

    side = nums[0]
    p = 4 * side

    return {
        "supported": True,
        "topic": "measurement",
        "method": "square_perimeter",
        "final": format_number(p),
        "working": (
            f"Step 1: Perimeter of square = 4 × side.\n"
            f"Step 2: 4 × {format_number(side)} = {format_number(p)}.\n"
            f"Answer: {format_number(p)}"
        ),
        "why": "A square has 4 equal sides, so multiply one side by 4.",
    }


def solve_volume(nums):
    if len(nums) < 3:
        return unsupported("Need length, width, and height.", "measurement", "volume")

    length, width, height = nums[0], nums[1], nums[2]
    v = length * width * height

    return {
        "supported": True,
        "topic": "measurement",
        "method": "volume",
        "final": format_number(v),
        "working": (
            f"Step 1: Volume = length × width × height.\n"
            f"Step 2: {format_number(length)} × {format_number(width)} × {format_number(height)} = {format_number(v)}.\n"
            f"Answer: {format_number(v)}"
        ),
        "why": "Volume measures how much space a 3D object takes up.",
    }
