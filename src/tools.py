"""
Utility tools for hybrid RAG + Agent workflows.

This module intentionally keeps tools small and safe so they can be
used in a constrained agent loop without arbitrary code execution.
"""

import ast
import math
import operator
import re
from typing import Optional

# Safe arithmetic operators allowed in expressions.
_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_CALLS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}


class UnsafeExpressionError(ValueError):
    """Raised when a calculator expression contains unsafe syntax."""


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.Name) and node.id in _ALLOWED_CONSTS:
        return _ALLOWED_CONSTS[node.id]

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        operand = _eval_ast(node.operand)
        return _ALLOWED_UNARY_OPS[type(node.op)](operand)

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name not in _ALLOWED_CALLS:
            raise UnsafeExpressionError(f"Unsupported function: {func_name}")
        args = [_eval_ast(arg) for arg in node.args]
        return _ALLOWED_CALLS[func_name](*args)

    raise UnsafeExpressionError("Unsupported or unsafe expression")


def calculator(expression: str) -> str:
    """Safely evaluate a math expression and return the result as text."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_ast(parsed)
        if isinstance(result, float):
            # Keep answers neat for student display (e.g., 42.0 -> 42)
            if abs(result - round(result)) < 1e-12:
                return str(int(round(result)))
            return f"{result:.10g}"
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def extract_first_numeric(text: str) -> Optional[float]:
    """Extract first numeric value from text for lightweight answer checking."""
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def is_calculation_heavy(question: str) -> bool:
    """Heuristic router for when to enable tool-augmented answering."""
    q = (question or "").lower()
    if re.search(r"\d+\s*[-+*/x×÷]\s*\d+", q):
        return True

    keywords = [
        "calculate", "how much", "how many", "total", "difference",
        "sum", "product", "quotient", "percentage", "ratio", "mean",
        "average", "per", "cost", "discount", "increase", "decrease",
    ]
    hit_count = sum(1 for kw in keywords if kw in q)
    has_number = bool(re.search(r"\d", q))
    return has_number and hit_count >= 1


TOOLS = {
    "calculator": {
        "function": calculator,
        "description": "Evaluate arithmetic expression safely. Example input: '1847*293+5812'",
    }
}
