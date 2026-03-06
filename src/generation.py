import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.retrieval import get_retriever

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

PROMPT_TEMPLATE = """\
You are a helpful and patient PSLE Math tutor for Primary 5 and Primary 6 students in Singapore.

Your job:
1. Read the retrieved reference examples below.
2. Use them to answer the student's question with clear, step-by-step working.
3. Explain each step in simple language that a 11-12 year old can understand.
4. At the end of your answer, cite which retrieved example(s) helped you by listing their numbers and sources.

Retrieved examples:
{context}

Student's question:
{question}

Answer (show step-by-step working, then cite sources):
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
chain = prompt | llm


def format_context(docs):
    """Format a list of LangChain Documents into a numbered string with source metadata."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str):
    """Retrieve relevant documents, build context, and generate a grounded answer."""
    retriever = get_retriever(k=4)
    docs = retriever.invoke(question)

    context = format_context(docs)
    sources = [doc.metadata.get("source", "unknown") for doc in docs]

    result = chain.invoke({"context": context, "question": question})
    answer = result.content

    print(f"[generation] Answered question using {len(docs)} retrieved documents.")
    return {"answer": answer, "sources": sources}
