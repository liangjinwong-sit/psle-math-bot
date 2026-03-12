import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval import retrieve_by_topic, retrieve_with_scores
from src.topic_classifier import get_topic_display_name

load_dotenv()

# Confidence threshold: if best retrieval similarity is below this,
# the bot warns the user that it may not have good reference material.
CONFIDENCE_THRESHOLD = 0.35

# Lazy-initialized LLM instance (avoids crash at import time if API key is missing)
_llm_instance = None


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
            model="gemini-2.5-flash",
            temperature=0.2,
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
        parts.append(f"[Example {i}] (Topic: {topic})\n{doc.page_content}")
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
        # Extract the solution method (first 2 lines of solution)
        content = doc.page_content
        method_snippet = ""
        if "Solution:" in content:
            solution_part = content.split("Solution:")[1].split("Final Answer:")[0].strip()
            # Take first 2 lines of solution as method snippet
            solution_lines = [line.strip() for line in solution_part.split("\n") if line.strip()]
            method_snippet = " → ".join(solution_lines[:2])
            if len(method_snippet) > 150:
                method_snippet = method_snippet[:147] + "..."
        
        citations.append({
            "topic": get_topic_display_name(doc.metadata.get("topic", "general")),
            "question": doc.metadata.get("question", "")[:120],
            "method": method_snippet,
            "score": round(score, 3),
            "source": f"GSM8K #{doc.metadata.get('id', '?')}",
        })
    return citations


def answer_question(question: str, topic: str = None, k: int = 4):
    """
    Main RAG pipeline: retrieve relevant examples and generate a tutoring response.
    
    Args:
        question: Student's math question
        topic: Optional PSLE topic key to filter retrieval (e.g., "percentage")
        k: Number of example problems to retrieve (default: 4)
    
    Returns:
        dict with keys: answer, citations, topic, topic_display,
        num_docs_retrieved, confidence, low_confidence
    """
    # Step 1: Retrieve relevant examples (with scores)
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
        }

    # Step 2: Check retrieval confidence
    best_score = max(score for _, score in docs_with_scores)
    avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
    low_confidence = best_score < CONFIDENCE_THRESHOLD

    # Step 3: Generate answer using LLM
    context = format_context(docs_with_scores)
    citations = format_citations(docs_with_scores)

    try:
        llm = _get_llm()
        chain = prompt | llm
        result = chain.invoke({
            "context": context,
            "question": question,
            "topic": topic_display,
        })
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
        }

    # Step 4: Add confidence warning if needed
    if low_confidence:
        answer = (
            "⚠️ **Note:** The examples I found may not be closely related to your question. "
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
    }
