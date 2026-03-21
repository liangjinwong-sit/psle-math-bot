# ============================================================
# PASTE THIS INTO src/generation.py
# ============================================================
# This file contains the exact code blocks to replace in
# generation.py. Follow the 3 steps below.
# ============================================================
import os
import re
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval import retrieve_by_topic, retrieve_with_scores
from src.tools import TOOLS, is_calculation_heavy
from src.topic_classifier import get_topic_display_name, is_math_question

load_dotenv()

# LLM provider: "gemini" (default), "openai", "groq", or "ollama"
# Override via LLM_PROVIDER in .env or --provider flag in evaluate.py
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Gemini settings (default — free tier, cloud)
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.2

# OpenAI settings (proprietary, cloud, paid)
OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.2

# Groq settings (open-source Llama, cloud, free)
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
GROQ_TEMPERATURE = 0.2

# Ollama settings (open-source Llama, local, free)
OLLAMA_MODEL_NAME = "llama3.1:8b"
OLLAMA_TEMPERATURE = 0.2
OLLAMA_BASE_URL = "http://localhost:11434"

# ---


# ── STEP 2 ───────────────────────────────────────────────────
# Find the existing _get_llm() function (and _llm_instance).
# Delete from the line:
#   _llm_instance = None
# all the way through the end of _get_llm().
#
# Replace with EVERYTHING between the --- markers below:
# ---

_llm_instance = None
_llm_provider_used = None


def _get_llm(provider: str = None):
    """
    Lazily initialize the LLM. Supports multiple providers.

    Providers:
        - "gemini": Google Gemini 2.5 Flash (default, free tier)
        - "openai": OpenAI GPT-4o-mini (paid)
        - "groq":   Llama 3.1 8B via Groq Cloud (free, open-source model)
        - "ollama": Llama 3.1 8B via local Ollama (free, runs on your machine)

    Args:
        provider: Provider name. If None, uses LLM_PROVIDER env/config.

    Returns:
        LangChain ChatModel instance
    """
    global _llm_instance, _llm_provider_used

    target = provider or LLM_PROVIDER

    # Reuse cached instance if same provider
    if _llm_instance is not None and _llm_provider_used == target:
        return _llm_instance

    if target == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY not set. Please add it to your .env file.\n"
                "Get your free API key at: https://aistudio.google.com/app/apikey"
            )
        _llm_instance = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=GEMINI_TEMPERATURE,
            google_api_key=api_key,
        )
        _llm_provider_used = "gemini"

    elif target == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY not set. Please add it to your .env file.")
        from langchain_openai import ChatOpenAI
        _llm_instance = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            temperature=OPENAI_TEMPERATURE,
            api_key=api_key,
        )
        _llm_provider_used = "openai"

    elif target == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY not set. Please add it to your .env file.\n"
                "Get your free key at: https://console.groq.com/keys"
            )
        from langchain_groq import ChatGroq
        _llm_instance = ChatGroq(
            model=GROQ_MODEL_NAME,
            temperature=GROQ_TEMPERATURE,
            api_key=api_key,
        )
        _llm_provider_used = "groq"

    elif target == "ollama":
        from langchain_ollama import ChatOllama
        _llm_instance = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            temperature=OLLAMA_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
        )
        _llm_provider_used = "ollama"

    else:
        raise ValueError(
            f"Unknown LLM provider: '{target}'. "
            f"Use 'gemini', 'openai', 'groq', or 'ollama'."
        )

    return _llm_instance


def get_current_provider() -> str:
    """Return the name of the currently active LLM provider."""
    return _llm_provider_used or LLM_PROVIDER


def switch_provider(provider: str):
    """
    Switch the LLM provider. Clears the cached instance so the
    next call to _get_llm() creates a new client.
    """
    global _llm_instance, _llm_provider_used
    _llm_instance = None
    _llm_provider_used = None
    os.environ["LLM_PROVIDER"] = provider

# ---


# ── STEP 3 ───────────────────────────────────────────────────
# In src/evaluate.py, make these changes:
#
# A) Add --provider to argparse (in __main__ block):
#
#    parser.add_argument(
#        "--provider", type=str, default=None,
#        choices=["gemini", "openai", "groq", "ollama"],
#        help="LLM provider for answer generation",
#    )
#
# B) Change run_evaluation call to:
#
#    run_evaluation(quick=args.quick, topic_filter=args.topic, provider=args.provider)
#
# C) Change run_evaluation signature to:
#
#    def run_evaluation(quick=False, topic_filter=None, provider=None):
#
# D) Right after "questions = BENCHMARK_QUESTIONS", add:
#
#    if provider:
#        from src.generation import switch_provider
#        switch_provider(provider)
#        print(f"  LLM provider: {provider}")
#
# E) In the summary section, after "Benchmark size:", add:
#
#    print(f"  LLM provider: {provider or 'gemini'}")
#
# F) In the results dict, add:
#
#    "provider": provider or "gemini",
