from src.solvers.rate import solve_rate
from src.solvers.percentage import solve_percentage
from src.solvers.ratio_proportion import solve_ratio_proportion
from src.solvers.measurement import solve_measurement
from src.solvers.data_handling import solve_data_handling
from src.solvers.fractions_decimals import solve_fractions_decimals


def solve_question_by_route(question: str, route: dict):
    topic = route.get("topic")

    if topic == "rate":
        return solve_rate(question, route)

    if topic == "percentage":
        return solve_percentage(question, route)

    if topic == "ratio_proportion":
        return solve_ratio_proportion(question, route)

    if topic == "measurement":
        return solve_measurement(question, route)

    if topic == "data_handling":
        return solve_data_handling(question, route)

    if topic == "fractions_decimals":
        return solve_fractions_decimals(question, route)

    return {
        "supported": False,
        "topic": topic,
        "method": route.get("method"),
        "final": None,
        "working": "No solver is available for this route.",
        "why": None,
    }
