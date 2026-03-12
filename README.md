# PSLE Math Study Bot

A RAG-powered math tutoring chatbot for Primary 5–6 students, built with the GSM8K dataset, FAISS, and Google Gemini.

## Features

- **RAG-Powered Q&A** — Retrieves similar worked examples from 7,473 GSM8K problems, then generates step-by-step solutions grounded in those examples.
- **Topic-Filtered Retrieval** — Questions are classified into 6 PSLE math topic families; retrieval can be filtered by topic for more relevant examples.
- **Structured Citations** — Each answer shows the retrieved reference problems with topic, similarity score, and solution method snippet.
- **Confidence-Based Fallback** — When retrieval similarity is low, the bot warns the user rather than presenting unreliable answers.
- **Practice Mode** — Pull random questions from the GSM8K test set, filtered by topic.
- **AI Question Generation** — LLM generates fresh practice questions on any topic at easy/medium/hard difficulty (extension feature).
- **Evaluation Suite** — Automated benchmark with 28 questions measuring topic classification accuracy, retrieval relevance, and answer correctness.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Dataset | GSM8K (Grade School Math 8K) — 7,473 training examples |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) — runs locally |
| Vector Store | FAISS (local, no external dependencies) |
| LLM | Google Gemini (`gemini-2.0-flash`) — free tier available |
| Framework | LangChain for RAG pipeline orchestration |
| UI | Streamlit |

## Project Structure

```
psle-math-bot/
├── app.py                          # Streamlit entry point
├── build_index.py                  # Build FAISS index from GSM8K
├── requirements.txt                # Python dependencies (pinned)
├── .env.example                    # API key template
├── data/
│   └── benchmark/
│       └── benchmark_questions.md  # Human-readable benchmark set
├── src/
│   ├── __init__.py
│   ├── ingest.py                   # GSM8K dataset loading + topic tagging
│   ├── retrieval.py                # FAISS index building, loading, search (cached)
│   ├── generation.py               # Gemini LLM generation with RAG + citations
│   ├── topic_classifier.py         # Keyword-based PSLE topic classifier
│   ├── practice.py                 # Random questions + AI question generation
│   ├── evaluate.py                 # Evaluation suite (classification, retrieval, answers)
│   └── ui.py                       # Streamlit UI components
├── test_gemini_models.py           # Utility: check available Gemini models
└── test_topics.py                  # Utility: verify topic metadata in index
```

## Setup

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

Get a free API key at: https://aistudio.google.com/app/apikey

### 3. Build FAISS Index

```bash
python build_index.py
```

This downloads GSM8K and creates the vector index (~5–10 minutes on first run).

### 4. Run the App

```bash
streamlit run app.py
```

### 5. Run Evaluation (optional)

```bash
python -m src.evaluate              # Full evaluation (needs API key)
python -m src.evaluate --quick      # Classification + retrieval only (no API calls)
python -m src.evaluate --topic percentage  # Single topic
```

## RAG Architecture

```
Student Question
      │
      ▼
┌─────────────────────┐
│  Topic Classifier    │  ← keyword + regex classification into 6 PSLE topics
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Local Embeddings    │  ← sentence-transformers (all-MiniLM-L6-v2)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  FAISS Vector Search │  ← 7,473 indexed examples, optional topic filtering
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Confidence Check    │  ← similarity threshold → fallback warning if low
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Gemini LLM          │  ← generates step-by-step solution from retrieved context
└──────────┬──────────┘
           ▼
     Answer + Citations
```

## PSLE Topic Families

The bot covers 6 core PSLE math topic families:

1. **Fractions & Decimals** — operations, comparisons, conversions
2. **Percentage** — percentage of quantity, increase/decrease, discounts
3. **Ratio & Proportion** — sharing, equivalent ratios, scaling
4. **Rate / Unitary Reasoning** — unit rate, speed, multi-step
5. **Measurement** — area, perimeter, volume of standard shapes
6. **Data Handling** — mean/average, reading tables and graphs

## Dataset Attribution

This project uses the GSM8K dataset:

- **Paper**: "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)
- **Source**: OpenAI
- **License**: MIT
- **HuggingFace**: https://huggingface.co/datasets/openai/gsm8k
