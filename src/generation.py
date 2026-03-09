import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval import retrieve_by_topic, get_retriever
from src.topic_classifier import get_topic_display_name

load_dotenv()

# Check if API key is available
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
    print("⚠️  GOOGLE_API_KEY not set. Please add it to your .env file.")
    print("   Get your free API key at: https://aistudio.google.com/app/apikey")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY,
)

# Enhanced prompt for topic-specific math tutoring with RAG
PROMPT_TEMPLATE = """\
You are a helpful and patient PSLE Math tutor for Primary 5 and Primary 6 students in Singapore.

Topic: {topic}

Your role:
1. Read the retrieved example problems from the same topic carefully
2. Use similar approaches from the examples to solve the student's question
3. Show clear step-by-step working with explanations suitable for P5-P6 students
4. State the final answer clearly at the start
5. Explain the mathematical reasoning in simple terms an 11-12 year old can understand

Retrieved example problems from {topic}:
{context}

Student's question:
{question}

Please provide:
1. **Final Answer:** (state it clearly at the start)
2. **Step-by-Step Solution:** (show all working with explanations)
3. **Key Concept:** (what PSLE math concept or method was used)

Your response:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
chain = prompt | llm


def format_context(docs):
    """Format a list of LangChain Documents into a numbered string with source metadata."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        topic = doc.metadata.get("topic", "")
        parts.append(f"[Example {i}] (Topic: {topic})\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str, topic: str = None, k: int = 4):
    """
    Retrieve relevant example problems and generate a tutoring response.
    
    Args:
        question: Student's math question
        topic: Optional PSLE topic to filter examples (e.g., "percentage", "rate")
        k: Number of example problems to retrieve (default: 4)
    
    Returns:
        dict with answer, sources, topic info, and metadata
    """
    # If topic is specified, use topic-filtered retrieval
    if topic:
        docs = retrieve_by_topic(question, topic, k)
        topic_display = get_topic_display_name(topic)
        print(f"[generation] Retrieved {len(docs)} examples from topic: {topic_display}")
    else:
        # Use general retrieval
        retriever = get_retriever(k=k)
        docs = retriever.invoke(question)
        topic_display = "All Topics"
        print(f"[generation] Retrieved {len(docs)} examples from all topics")

    if not docs:
        return {
            "answer": f"Sorry, I couldn't find relevant examples for this question in the {topic_display} topic. Try asking a different question or selecting a different topic.",
            "sources": [],
            "topic": topic,
            "num_docs_retrieved": 0,
        }

    context = format_context(docs)
    sources = [
        {
            "question": doc.metadata.get("question", "")[:100] + "...",
            "topic": doc.metadata.get("topic", ""),
        }
        for doc in docs
    ]

    result = chain.invoke({
        "context": context,
        "question": question,
        "topic": topic_display,
    })
    answer = result.content

    return {
        "answer": answer,
        "sources": sources,
        "topic": topic,
        "topic_display": topic_display,
        "num_docs_retrieved": len(docs),
    }
