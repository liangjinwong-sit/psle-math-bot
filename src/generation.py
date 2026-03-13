import os
import re
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval import retrieve_by_topic, retrieve_with_scores
from src.tools import TOOLS, is_calculation_heavy
from src.topic_classifier import get_topic_display_name

load_dotenv()

# ── Configuration Constants ──────────────────────────────────────────
# These control key behaviours across the RAG pipeline.

# Confidence threshold: if the best retrieval similarity score falls
# below this value, the system switches to the fallback prompt that
# relies on the LLM's own knowledge instead of potentially irrelevant
# retrieved examples.  Tuned empirically on the benchmark set.
CONFIDENCE_THRESHOLD = 0.35

# Number of similar examples to retrieve from FAISS for each query.
# 4 gives enough context without overloading the prompt token budget.
DEFAULT_RETRIEVAL_K = 4

# Maximum characters to show per citation method snippet in the UI.
CITATION_SNIPPET_MAX_LEN = 150

# Citation questions should be shown in full so students can inspect
# the complete source question without mid-sentence truncation.

# Number of progressive hints generated per question.
MAX_HINTS = 3

# LLM model name and temperature for all generation tasks.
LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2

# Retry policy for temporary LLM/API errors.
MAX_LLM_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0

# Hybrid RAG+Agent tool settings.
ENABLE_TOOL_AGENT = True
TOOL_AGENT_MAX_STEPS = 3

# Retrieved context is untrusted text. Remove lines that look like
# instruction injection attempts before passing to the model.
CONTEXT_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"follow\s+these\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"assistant\s*:",
    r"user\s*:",
]

# Lazy-initialized LLM instance (avoids crash at import time if API key is missing)
_llm_instance = None


def _strip_injection_lines(text: str) -> str:
    """Remove suspicious instruction-like lines from retrieved context."""
    safe_lines = []
    for raw_line in (text or "").split("\n"):
        line = raw_line.strip()
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in CONTEXT_INJECTION_PATTERNS):
            continue
        safe_lines.append(raw_line)
    return "\n".join(safe_lines).strip()


def _invoke_with_retries(invoke_fn):
    """Execute LLM invoke call with short exponential backoff on transient errors."""
    last_error = None
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            return invoke_fn()
        except ValueError:
            # Preserve explicit config errors (e.g., missing API key).
            raise
        except Exception as e:
            last_error = e
            if attempt < MAX_LLM_RETRIES:
                sleep_seconds = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(sleep_seconds)
    raise last_error


def _get_llm():
    """Lazily initialize the Gemini LLM. Only creates the client when first needed."""
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY not set. Please add it to your .env file.\n"
                "Get your free API key at: https://aistudio.google.com/app/apikey"
            )
        _llm_instance = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            google_api_key=api_key,
        )
    return _llm_instance


# Prompt template for topic-specific math tutoring with RAG
PROMPT_TEMPLATE = """\
You are a helpful and patient PSLE Math tutor for Primary 5 and Primary 6 students in Singapore.

Topic: {topic}

Your role:
1. Read the retrieved example problems from the same topic carefully
2. Use similar solution approaches from the examples to solve the student's question
3. Show clear step-by-step working with explanations suitable for P5-P6 students
4. State the final answer clearly
5. Explain the mathematical reasoning in simple terms an 11-12 year old can understand

Retrieved example problems from {topic}:
{context}

Student's question:
{question}

Please provide your answer in this format:
1. **Final Answer:** State the answer clearly
2. **Step-by-Step Solution:** Show all working with explanations
3. **Key Concept:** Name the PSLE math concept or method used (e.g., "unitary method", "percentage of a quantity")

Your response:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# Fallback prompt when retrieval confidence is too low
FALLBACK_PROMPT_TEMPLATE = """\
You are a helpful and patient PSLE Math tutor for Primary 5 and Primary 6 students in Singapore.

Topic: {topic}

I could not find closely matching example problems in my database, so please answer using your own mathematical knowledge.

Student's question:
{question}

Please provide your answer in this format:
1. **Final Answer:** State the answer clearly
2. **Step-by-Step Solution:** Show all working with explanations
3. **Key Concept:** Name the PSLE math concept or method used (e.g., "unitary method", "percentage of a quantity")

Your response:
"""

fallback_prompt = ChatPromptTemplate.from_template(FALLBACK_PROMPT_TEMPLATE)


def _build_tool_agent_prompt(question: str, topic: str, context: str, scratchpad: str) -> str:
    """Build a constrained ReAct-style prompt for the tool agent loop."""
    tool_lines = []
    for name, info in TOOLS.items():
        tool_lines.append(f"- {name}: {info['description']}")
    tool_catalog = "\n".join(tool_lines)

    return (
        "You are a PSLE Math tutor with tools. Use retrieved context when useful.\n\n"
        f"Topic: {topic}\n"
        f"Question: {question}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        "Available tools:\n"
        f"{tool_catalog}\n\n"
        "Reply using EXACTLY one of these formats:\n"
        "THINK: <reasoning>\n"
        "ACT: <tool_name>(<tool_input>)\n\n"
        "OR\n"
        "THINK: <reasoning>\n"
        "ANSWER: <final answer for student>\n\n"
        "Rules:\n"
        "1. Use one tool at a time.\n"
        "2. If you use ACT, tool_input must be plain text without markdown.\n"
        "3. If enough information is available, return ANSWER.\n\n"
        "Conversation history:\n"
        f"{scratchpad}"
    )


def _parse_tool_agent_output(text: str) -> dict:
    """Parse THINK/ACT/ANSWER output from the tool agent."""
    content = (text or "").strip()

    think_match = re.search(r"THINK:\s*(.+?)(?=ACT:|ANSWER:|$)", content, re.DOTALL | re.IGNORECASE)
    think = think_match.group(1).strip() if think_match else ""

    answer_match = re.search(r"ANSWER:\s*(.+)", content, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return {
            "type": "final_answer",
            "think": think,
            "answer": answer_match.group(1).strip(),
        }

    act_match = re.search(r"ACT:\s*(\w+)\((.*)\)", content, re.DOTALL | re.IGNORECASE)
    if act_match:
        tool_name = act_match.group(1).strip()
        tool_input = act_match.group(2).strip().strip("\"'")
        return {
            "type": "tool_call",
            "think": think,
            "tool_name": tool_name,
            "tool_input": tool_input,
        }

    # If format is malformed, treat it as final answer text.
    return {
        "type": "final_answer",
        "think": think,
        "answer": content,
    }


def _execute_tool(tool_name: str, tool_input: str) -> str:
    """Execute a registered tool safely."""
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'. Available tools: {list(TOOLS.keys())}"
    func = TOOLS[tool_name]["function"]
    return func(tool_input)


def _run_tool_agent_with_context(question: str, topic: str, context: str):
    """Run a short ReAct-style loop.

    Returns:
        tuple[str, list[str]]: (answer_text_or_empty, used_tool_names)
    """
    llm = _get_llm()
    scratchpad = f"User Question: {question}\n\n"
    used_tools = []

    for _ in range(TOOL_AGENT_MAX_STEPS):
        prompt_text = _build_tool_agent_prompt(question, topic, context, scratchpad)
        result = _invoke_with_retries(lambda: llm.invoke(prompt_text))
        response_text = result.content.strip()
        parsed = _parse_tool_agent_output(response_text)

        if parsed["type"] == "final_answer":
            return parsed["answer"], used_tools

        tool_name = parsed["tool_name"]
        tool_input = parsed["tool_input"]
        if tool_name not in used_tools:
            used_tools.append(tool_name)
        tool_result = _execute_tool(tool_name, tool_input)
        scratchpad += f"{response_text}\nOBSERVE: {tool_result}\n\n"

    return "", used_tools


def format_context(docs_with_scores):
    """Format retrieved documents into numbered context string for the LLM prompt.
    
    Args:
        docs_with_scores: List of (Document, similarity_score) tuples
    
    Returns:
        Formatted string with numbered examples and their topics
    """
    parts = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        topic = doc.metadata.get("topic", "")
        safe_content = _strip_injection_lines(doc.page_content)
        parts.append(f"[Example {i}] (Topic: {topic})\n{safe_content}")
    return "\n\n".join(parts)


def format_citations(docs_with_scores):
    """Format retrieved documents into structured citation objects for the UI.
    
    Each citation includes the topic, a preview of the example question,
    the solution method snippet, and the retrieval similarity score.
    
    Args:
        docs_with_scores: List of (Document, similarity_score) tuples
    
    Returns:
        List of citation dicts with topic, question, method, and score
    """
    citations = []
    for doc, score in docs_with_scores:
        # Each GSM8K document has the format:
        #   "Question: ... Solution: ... Final Answer: ..."
        # We extract the first two solution lines as a method preview.
        content = doc.page_content
        method_snippet = ""
        if "Solution:" in content:
            # Isolate the solution text between "Solution:" and "Final Answer:"
            solution_part = content.split("Solution:")[1].split("Final Answer:")[0].strip()
            # Keep only the first 2 non-empty lines as a brief preview
            solution_lines = [line.strip() for line in solution_part.split("\n") if line.strip()]
            raw_snippet = " -> ".join(solution_lines[:2])
            # Strip <<calc>> annotations that GSM8K uses internally (e.g. <<40/2=20>>)
            raw_snippet = re.sub(r"<<[^>]*>>", "", raw_snippet).strip()
            # Truncate to keep the citation compact
            if len(raw_snippet) > CITATION_SNIPPET_MAX_LEN:
                method_snippet = raw_snippet[:CITATION_SNIPPET_MAX_LEN] + "..."
            else:
                method_snippet = raw_snippet

        citations.append({
            "topic": get_topic_display_name(doc.metadata.get("topic", "general")),
            "question": doc.metadata.get("question", ""),
            "method": method_snippet,
            "score": round(score, 3),
            "source": f"GSM8K #{doc.metadata.get('id', '?')}",
        })
    return citations


def auto_mark_answer(question: str, correct_answer: str, student_answer: str) -> dict:
    """
    Use Gemini to auto-mark a student's answer and provide feedback.

    Args:
        question: The math question text
        correct_answer: The correct answer (from GSM8K or generated)
        student_answer: The student's submitted answer

    Returns:
        dict with is_correct, score, feedback
    """
    llm = _get_llm()

    mark_prompt = (
        "You are a PSLE Math marker for Primary 5-6 students in Singapore.\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct_answer}\n"
        f"Student's Answer: {student_answer}\n\n"
        "Compare the student's answer to the correct answer. The student may express the answer "
        "differently (e.g., '$12' vs '12', '3/5' vs '0.6') — accept equivalent values.\n\n"
        "Respond in EXACTLY this format:\n"
        "VERDICT: CORRECT or INCORRECT\n"
        "FEEDBACK: [Brief encouraging feedback for the student. "
        "If incorrect, explain the mistake and hint at the right approach.]"
    )

    try:
        result = _invoke_with_retries(lambda: llm.invoke(mark_prompt))
        response = result.content.strip()

        # Default to INCORRECT so that a malformed LLM response is
        # treated as a failed mark rather than a false positive.
        verdict = "INCORRECT"
        feedback = response

        # Parse the structured VERDICT/FEEDBACK format from the LLM.
        # The LLM occasionally adds extra text, so we scan line-by-line.
        for line in response.split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("VERDICT:"):
                verdict = line_stripped[len("VERDICT:"):].strip().upper()
            elif line_stripped.startswith("FEEDBACK:"):
                feedback = line_stripped[len("FEEDBACK:"):].strip()

        # "CORRECT" must appear without "INCORRECT" prefix to count as correct
        is_correct = "CORRECT" in verdict and "INCORRECT" not in verdict

        return {
            "is_correct": is_correct,
            "score": 1 if is_correct else 0,
            "feedback": feedback,
        }
    except Exception as e:
        return {
            "is_correct": False,
            "score": 0,
            "feedback": f"Could not auto-mark: {str(e)}",
        }


def explain_mistake(question: str, correct_answer: str, student_answer: str) -> str:
    """
    Generate a detailed, kid-friendly explanation of what the student did wrong.

    Uses simple language a Primary 5-6 student can understand.
    """
    llm = _get_llm()

    prompt_text = (
        "You are a kind and patient PSLE Math tutor helping an 11-12 year old student "
        "in Singapore understand their mistake.\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct_answer}\n"
        f"Student's Answer: {student_answer}\n\n"
        "The student got this wrong. Please explain in plain English with NO LaTeX, "
        "NO dollar signs used as math symbols, and NO backslash commands like \\times or \\frac. "
        "Write all math as plain text, e.g. '4000 x 2 = 8000' or '5000 + 4000 + 8000 = 17000'.\n\n"
        "Format your reply using these exact headings:\n"
        "### 1. What went wrong\n"
        "[Explain in 2-3 short simple sentences what mistake the student likely made.]\n\n"
        "### 2. How to solve it step by step\n"
        "[Walk through the correct solution as a numbered list. "
        "Each step on its own line. Short sentences. Maximum 5 steps.]\n\n"
        "### 3. Quick tip to remember\n"
        "[One sentence. Easy to remember.]\n\n"
        "Keep the tone warm and encouraging. The student is 11-12 years old."
    )

    try:
        result = _invoke_with_retries(lambda: llm.invoke(prompt_text))
        return result.content.strip()
    except Exception as e:
        return f"Sorry, I couldn't generate an explanation right now: {str(e)}"


def generate_hints(question: str, correct_answer: str) -> list:
    """
    Generate 3 progressive hints for a math question.

    Hint 1: Identify what the question is asking (gentle nudge)
    Hint 2: Suggest the method/concept to use
    Hint 3: Give a partial working that makes the next step obvious
    """
    llm = _get_llm()

    prompt_text = (
        "You are a PSLE Math tutor helping a Primary 5-6 student in Singapore.\n\n"
        f"Question: {question}\n"
        f"Answer: {correct_answer}\n\n"
        "The student is stuck. Generate exactly 3 hints, from easiest to most helpful. "
        "Each hint should help them a little bit more, but NEVER reveal the final answer.\n\n"
        "Use simple English that an 11-12 year old can understand. Keep each hint to 1-2 sentences.\n\n"
        "Respond in EXACTLY this format:\n"
        "HINT1: [What is the question really asking? Help them understand the problem.]\n"
        "HINT2: [What method or concept should they use? e.g. 'Try using the unitary method']\n"
        "HINT3: [Give a partial calculation or setup, e.g. 'First find: 20% of 60 = ...']\n"
    )

    try:
        result = _invoke_with_retries(lambda: llm.invoke(prompt_text))
        response = result.content.strip()

        # Parse each HINT tag from the LLM output.  We scan every line
        # because the model sometimes adds blank lines between hints.
        hints = []
        for line in response.split("\n"):
            line = line.strip()
            for tag in ["HINT1:", "HINT2:", "HINT3:"]:
                if line.upper().startswith(tag):
                    hints.append(line[len(tag):].strip())

        # Safety fallback: pad to MAX_HINTS if the LLM returned fewer
        while len(hints) < MAX_HINTS:
            hints.append("Try reading the question again carefully and think about what operation to use.")

        return hints[:MAX_HINTS]
    except Exception as e:
        return [
            "Read the question carefully. What is it asking you to find?",
            "Think about which math operation (add, subtract, multiply, divide) you need.",
            f"Could not generate more hints: {str(e)}",
        ]


def generate_mcq_options(question: str, correct_answer: str) -> list:
    """
    Generate 4 MCQ options (1 correct + 3 plausible distractors) for a question.

    Returns a shuffled list of {"label": "A"/"B"/"C"/"D", "text": ..., "is_correct": bool}.
    """
    llm = _get_llm()

    prompt_text = (
        "You are a PSLE Math question writer for Primary 5-6 students in Singapore.\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct_answer}\n\n"
        "Generate 3 wrong but plausible answer choices that a student might pick if they "
        "made a common mistake (e.g. wrong operation, forgot a step, calculation error).\n\n"
        "Respond in EXACTLY this format (numbers/values only, no explanation):\n"
        "WRONG1: [first wrong answer]\n"
        "WRONG2: [second wrong answer]\n"
        "WRONG3: [third wrong answer]\n"
    )

    try:
        result = _invoke_with_retries(lambda: llm.invoke(prompt_text))
        response = result.content.strip()

        # Parse 3 distractor answers from the structured LLM response
        wrong_answers = []
        for line in response.split("\n"):
            line = line.strip()
            for tag in ["WRONG1:", "WRONG2:", "WRONG3:"]:
                if line.upper().startswith(tag):
                    wrong_answers.append(line[len(tag):].strip())

        # Ensure we always have exactly 3 distractors (pad if LLM gave fewer)
        while len(wrong_answers) < 3:
            wrong_answers.append("N/A")

        # Combine correct + distractors, shuffle so the correct answer
        # position is randomised (prevents students from learning "always pick A").
        import random
        options = [
            {"text": correct_answer, "is_correct": True},
            {"text": wrong_answers[0], "is_correct": False},
            {"text": wrong_answers[1], "is_correct": False},
            {"text": wrong_answers[2], "is_correct": False},
        ]
        random.shuffle(options)

        # Assign A/B/C/D labels after shuffling so labels match the new order
        labels = ["A", "B", "C", "D"]
        for i, opt in enumerate(options):
            opt["label"] = labels[i]

        return options
    except Exception as e:
        return [
            {"label": "A", "text": correct_answer, "is_correct": True},
            {"label": "B", "text": "Error generating options", "is_correct": False},
            {"label": "C", "text": "Error generating options", "is_correct": False},
            {"label": "D", "text": "Error generating options", "is_correct": False},
        ]


def generate_similar_question(question: str, topic: str, difficulty: str = "medium") -> dict:
    """
    Generate a similar question to the one the student got wrong, for retry practice.
    """
    llm = _get_llm()
    topic_display = get_topic_display_name(topic) if topic else "General"

    prompt_text = (
        "You are a PSLE Math question writer for Primary 5-6 students in Singapore.\n\n"
        f"The student just attempted this question and wants to practice a similar one:\n"
        f"Original Question: {question}\n"
        f"Topic: {topic_display}\n"
        f"Difficulty: {difficulty}\n\n"
        "Generate ONE new question that tests the SAME concept and uses a SIMILAR structure, "
        "but with DIFFERENT numbers and context. "
        "Use Singapore context (names like Ali, Mei Ling, Raju; SGD currency).\n\n"
        "Respond in EXACTLY this format:\n"
        "QUESTION: [Your similar question]\n"
        "SOLUTION: [Step-by-step solution]\n"
        "ANSWER: [Final numeric answer only]\n"
    )

    try:
        result = _invoke_with_retries(lambda: llm.invoke(prompt_text))
        response = result.content.strip()

        new_question = ""
        solution = ""
        answer = ""

        # Strategy 1 (index-based): works for multi-line SOLUTION blocks
        # because we slice between known markers instead of parsing line-by-line.
        if "SOLUTION:" in response and "ANSWER:" in response:
            q_start = response.find("QUESTION:") + len("QUESTION:") if "QUESTION:" in response else 0
            s_start = response.find("SOLUTION:")
            a_start = response.find("ANSWER:")
            new_question = response[q_start:s_start].strip()
            solution = response[s_start + len("SOLUTION:"):a_start].strip()
            answer = response[a_start + len("ANSWER:"):].strip()
        else:
            # Strategy 2 (line-by-line fallback): for simpler single-line responses
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("QUESTION:"):
                    new_question = line[len("QUESTION:"):].strip()
                elif line.startswith("SOLUTION:"):
                    solution = line[len("SOLUTION:"):].strip()
                elif line.startswith("ANSWER:"):
                    answer = line[len("ANSWER:"):].strip()

        return {
            "question": new_question or "Could not generate a similar question.",
            "solution": solution or "Solution not available.",
            "answer": answer or "N/A",
            "topic": topic,
            "topic_display": topic_display,
            "difficulty": difficulty,
        }
    except Exception as e:
        return {
            "question": f"Error: {str(e)}",
            "solution": "",
            "answer": "",
            "topic": topic,
            "topic_display": topic_display,
            "difficulty": difficulty,
        }


def answer_question(question: str, topic: str = None, k: int = DEFAULT_RETRIEVAL_K):
    """
    Main RAG pipeline: retrieve relevant examples and generate a tutoring response.
    
    Args:
        question: Student's math question
        topic: Optional PSLE topic key to filter retrieval (e.g., "percentage")
        k: Number of example problems to retrieve (default: 4)
    
    Returns:
        dict with keys: answer, citations, topic, topic_display,
        num_docs_retrieved, confidence, low_confidence, used_tools
    """
    # Step 1: Retrieve the k most similar solved problems from FAISS.
    # If a specific topic is given we do filtered retrieval; otherwise
    # we search across all topics.
    try:
        if topic:
            docs_with_scores = retrieve_by_topic(question, topic, k)
            topic_display = get_topic_display_name(topic)
        else:
            docs_with_scores = retrieve_with_scores(question, k)
            topic_display = "All Topics"
    except Exception as e:
        return {
            "answer": f"Error loading the knowledge base: {str(e)}\n\nPlease make sure the FAISS index has been built by running `python build_index.py`.",
            "citations": [],
            "topic": topic,
            "topic_display": topic_display if topic else "All Topics",
            "num_docs_retrieved": 0,
            "confidence": 0,
            "low_confidence": True,
            "used_tools": [],
        }

    if not docs_with_scores:
        return {
            "answer": f"Sorry, I couldn't find relevant examples for this question in the {topic_display} topic. Try selecting a different topic or rephrasing your question.",
            "citations": [],
            "topic": topic,
            "topic_display": topic_display,
            "num_docs_retrieved": 0,
            "confidence": 0,
            "low_confidence": True,
            "used_tools": [],
        }

    # Step 2: Check retrieval confidence.
    # best_score is the highest similarity among all retrieved docs.
    # If even the best match is below our threshold the context is
    # unlikely to help, so we switch to the fallback (LLM-only) prompt.
    best_score = max(score for _, score in docs_with_scores)
    avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
    low_confidence = best_score < CONFIDENCE_THRESHOLD

    # Step 3: Build the prompt context and citation metadata.
    context = format_context(docs_with_scores)
    citations = format_citations(docs_with_scores)

    try:
        answer = ""
        used_tools = []

        # Optional Lab6-style tool augmentation for arithmetic-heavy questions.
        # We keep it bounded and fall back to normal RAG if it cannot finish.
        if ENABLE_TOOL_AGENT and is_calculation_heavy(question):
            answer, used_tools = _run_tool_agent_with_context(
                question=question,
                topic=topic_display,
                context=context,
            )

        if not answer:
            llm = _get_llm()
            if low_confidence:
                # Fallback: skip unreliable retrieved context, use LLM knowledge
                chain = fallback_prompt | llm
                result = _invoke_with_retries(
                    lambda: chain.invoke({
                        "question": question,
                        "topic": topic_display,
                    })
                )
            else:
                chain = prompt | llm
                result = _invoke_with_retries(
                    lambda: chain.invoke({
                        "context": context,
                        "question": question,
                        "topic": topic_display,
                    })
                )
            answer = result.content
    except ValueError as e:
        # API key not set
        return {
            "answer": str(e),
            "citations": citations,
            "topic": topic,
            "topic_display": topic_display,
            "num_docs_retrieved": len(docs_with_scores),
            "confidence": best_score,
            "low_confidence": True,
            "used_tools": [],
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}\n\nThis may be a temporary API issue. Please try again.",
            "citations": citations,
            "topic": topic,
            "topic_display": topic_display,
            "num_docs_retrieved": len(docs_with_scores),
            "confidence": best_score,
            "low_confidence": True,
            "used_tools": [],
        }

    # Step 4: Add confidence warning if needed
    if low_confidence:
        answer = (
            "**Note:** The examples I found may not be closely related to your question. "
            "The answer below is my best attempt, but please verify the solution.\n\n"
            + answer
        )

    return {
        "answer": answer,
        "citations": citations,
        "topic": topic,
        "topic_display": topic_display,
        "num_docs_retrieved": len(docs_with_scores),
        "confidence": best_score,
        "low_confidence": low_confidence,
        "used_tools": used_tools,
    }
