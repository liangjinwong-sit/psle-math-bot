from src.router import route_question
from src.retrieval import load_index
from src.solvers import solve_question_by_route

UNSUPPORTED_MESSAGE = (
    "I'm sorry, but I can't answer this reliably with the current local solver and notes.\n\n"
    "Please try a clearer P5/P6 question from the supported PSLE topic families."
)

TOPIC_SOURCE_HINTS = {
    "fractions_decimals": ["fraction", "decimal", "fractions", "decimals"],
    "percentage": ["percentage", "percent"],
    "ratio_proportion": ["ratio", "proportion"],
    "rate": ["rate", "unit", "speed", "cost"],
    "measurement": ["measurement", "area", "perimeter", "volume"],
    "data_handling": ["data", "mean", "average", "graph", "table"],
}


def retrieve_supporting_docs(question: str, route: dict, k: int = 4):
    vectorstore = load_index()
    topic = route.get("topic", "") or ""
    method = route.get("method", "") or ""

    query = f"{topic} {method} {question}".strip()
    results = vectorstore.similarity_search_with_score(query, k=8)

    hints = TOPIC_SOURCE_HINTS.get(topic, [])
    filtered = []

    for doc, score in results:
        doc.metadata["score"] = float(score)
        source = doc.metadata.get("source", "").lower()
        content = doc.page_content.lower()

        if any(h in source or h in content for h in hints):
            filtered.append(doc)

    if not filtered:
        filtered = [doc for doc, _ in results]

    return filtered[:k]


def format_sources(docs):
    return [doc.metadata.get("source", "unknown") for doc in docs]


def format_supporting_notes(docs, max_chars=700):
    snippets = []

    for doc in docs[:2]:
        source = doc.metadata.get("source", "unknown")
        text = " ".join(doc.page_content.strip().split())

        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."

        snippets.append(f"Source: {source}\n{text}")

    return "\n\n".join(snippets)


def build_answer_text(route: dict, solver_result: dict, docs):
    parts = []

    parts.append(f"Detected topic: {route.get('topic')}")
    parts.append(f"Detected method: {route.get('method')}")

    if route.get("reason"):
        parts.append(f"Routing reason: {route.get('reason')}")

    if solver_result.get("final") is not None:
        parts.append(f"\nFinal answer:\n{solver_result['final']}")

    if solver_result.get("working"):
        parts.append(f"\nWorking:\n{solver_result['working']}")

    if solver_result.get("why"):
        parts.append(f"\nWhy this works:\n{solver_result['why']}")

    notes_text = format_supporting_notes(docs)
    if notes_text:
        parts.append(f"\nSupporting notes:\n{notes_text}")

    return "\n\n".join(parts)


def answer_question(question: str):
    route = route_question(question)

    if not route.get("topic"):
        return {
            "answer": UNSUPPORTED_MESSAGE,
            "sources": [],
            "supported": False,
            "mode": "unsupported",
            "route": route,
        }

    solver_result = solve_question_by_route(question, route)

    if not solver_result.get("supported", False):
        return {
            "answer": solver_result.get("working", UNSUPPORTED_MESSAGE),
            "sources": [],
            "supported": False,
            "mode": "unsupported",
            "route": route,
        }

    docs = retrieve_supporting_docs(question, route, k=4)
    sources = format_sources(docs)
    answer_text = build_answer_text(route, solver_result, docs)

    return {
        "answer": answer_text,
        "sources": sources,
        "supported": True,
        "mode": "solver",
        "route": route,
    }
