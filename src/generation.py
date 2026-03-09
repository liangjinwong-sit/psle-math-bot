import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval import get_retriever

load_dotenv()

# Check if API key is available
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
    print("⚠️  GOOGLE_API_KEY not set. Please add it to your .env file.")
    print("   Get your free API key at: https://aistudio.google.com/app/apikey")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY,
)

# Enhanced prompt for PSLE tutoring
PROMPT_TEMPLATE = """\
You are a patient, encouraging PSLE Math tutor for Primary 5 and Primary 6 students in Singapore.

Your teaching style:
1. Always start with the final answer clearly stated
2. Break down the solution into numbered steps
3. Use simple language that an 11-12 year old can understand
4. Show all working and calculations
5. Explain WHY each step is taken (the mathematical reasoning)
6. Use Singapore Math terminology and methods when relevant
7. Encourage the student and point out key concepts

Retrieved reference materials:
{context}

Student's question:
{question}

Please provide:
1. **Final Answer:** (state the answer clearly)
2. **Step-by-Step Working:** (numbered steps with explanations)
3. **Key Concept:** (what mathematical concept this uses)
4. **Reference Sources:** (mention which retrieved examples helped)

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
        topic_label = f" [Topic: {topic}]" if topic and topic != "general" else ""
        parts.append(f"[{i}] {source}{topic_label}\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str, k: int = 4):
    """
    Retrieve relevant documents, build context, and generate a tutoring response.
    
    Args:
        question: Student's PSLE math question
        k: Number of documents to retrieve (default: 4)
    
    Returns:
        dict with answer, sources, and metadata
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(question)

    context = format_context(docs)
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "topic": doc.metadata.get("topic", ""),
        }
        for doc in docs
    ]

    result = chain.invoke({"context": context, "question": question})
    answer = result.content

    print(f"[generation] Answered question using {len(docs)} retrieved documents.")
    
    return {
        "answer": answer,
        "sources": sources,
        "num_docs_retrieved": len(docs),
    }
