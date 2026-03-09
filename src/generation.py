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

# Enhanced prompt for math tutoring with RAG
PROMPT_TEMPLATE = """\
You are a helpful and patient math tutor for grade school students.

Your role:
1. Read the retrieved example problems and solutions below carefully
2. Use similar approaches from the examples to solve the student's question
3. Show clear step-by-step working with explanations
4. State the final answer clearly
5. Explain the mathematical reasoning in simple terms

Retrieved example problems:
{context}

Student's question:
{question}

Please provide:
1. **Final Answer:** (state it clearly at the start)
2. **Step-by-Step Solution:** (show all working with explanations)
3. **Key Insight:** (what math concept or method was used)

Your response:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
chain = prompt | llm


def format_context(docs):
    """Format a list of LangChain Documents into a numbered string with source metadata."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Example {i}] (Source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str, k: int = 4):
    """
    Retrieve relevant example problems from GSM8K and generate a tutoring response.
    
    Args:
        question: Student's math question
        k: Number of example problems to retrieve (default: 4)
    
    Returns:
        dict with answer, sources, and metadata
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(question)

    context = format_context(docs)
    sources = [doc.metadata.get("source", "unknown") for doc in docs]

    result = chain.invoke({"context": context, "question": question})
    answer = result.content

    print(f"[generation] Generated answer using {len(docs)} retrieved examples from GSM8K.")
    
    return {
        "answer": answer,
        "sources": sources,
        "num_docs_retrieved": len(docs),
    }
