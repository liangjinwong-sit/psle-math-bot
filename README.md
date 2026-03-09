# PSLE Math Study Bot

A Streamlit-based Math Study Bot that uses RAG (Retrieval-Augmented Generation) with the GSM8K dataset and Google Gemini LLM to help students learn mathematics through worked examples.

## Features
- **RAG-powered Q&A**: Retrieves similar problems from GSM8K dataset and generates step-by-step solutions
- **Gemini LLM Integration**: Uses Google Gemini for natural language explanations
- **Practice Mode**: Generate random practice questions with answer reveal
- **Source Citations**: Shows which example problems were used to generate the answer
- **Local Vector Search**: FAISS index for fast semantic similarity search

## Tech Stack
- **Dataset**: GSM8K (Grade School Math 8K) - 7,473 training examples
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2) - runs locally
- **Vector Store**: FAISS (local, no external dependencies)
- **LLM**: Google Gemini (gemini-1.5-flash) - free tier available
- **Framework**: LangChain for RAG pipeline orchestration
- **UI**: Streamlit

## RAG Architecture

```
Student Question
      ↓
[Local Embeddings] (sentence-transformers)
      ↓
[FAISS Vector Search] (8K+ indexed examples)
      ↓
Retrieve top-k similar problems
      ↓
[Gemini LLM] (with context from examples)
      ↓
Generate step-by-step solution
```

## Project Structure
```text
psle-math-bot/
├─ app.py                  # Main Streamlit application
├─ build_index.py          # Build FAISS index from GSM8K
├─ requirements.txt        # Python dependencies
├─ .env                    # API keys (create from .env.example)
├─ index/
│  └─ psle_faiss/         # FAISS vector store (generated)
└─ src/
   ├─ ingest.py           # Load GSM8K dataset
   ├─ retrieval.py        # FAISS retrieval logic
   ├─ generation.py       # Gemini LLM generation with RAG
   ├─ ui.py               # Streamlit UI components
   └─ practice.py         # Practice mode logic
```
## Installation

### 1. Create Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get a free API key at: [Google AI Studio](https://aistudio.google.com/app/apikey)

### 4. Build FAISS Index
```bash
python build_index.py
```

This downloads the GSM8K dataset and builds a FAISS vector index (~5-10 minutes).

Expected output:
```
[ingest] Loaded 7473 documents from GSM8K train split.
📊 Building FAISS index with 7473 documents...
✅ Index build complete!
```

### 5. Run the App
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## How It Works

### 1. **Data Source: GSM8K Dataset**
- Contains 8,500 high-quality grade school math word problems
- Each problem has step-by-step solutions
- Covers various math topics: arithmetic, percentages, ratios, etc.
- Publicly available via HuggingFace datasets

### 2. **RAG Pipeline**

**Indexing Phase** (run once with `build_index.py`):
1. Load GSM8K training set (7,473 examples)
2. Convert each problem+solution to embeddings using sentence-transformers
3. Store in FAISS vector index for fast similarity search

**Query Phase** (when student asks a question):
1. Convert question to embedding vector
2. Find top-k most similar problems from GSM8K
3. Pass question + retrieved examples to Gemini LLM
4. LLM generates step-by-step solution using similar approaches from examples

### 3. **Why This Approach Works**
- ✅ **Grounded Answers**: LLM uses actual worked examples, reducing hallucinations
- ✅ **Better Explanations**: Similar problems provide relevant solution patterns
- ✅ **Learning by Example**: Students see how similar problems were solved
- ✅ **No Manual Curation**: GSM8K provides 7,500+ high-quality examples
- ✅ **Free to Use**: All components use free tiers (Gemini, HuggingFace, FAISS)

## Usage

### Ask a Question
1. Type your math question in the text area
2. Click "Get Answer"
3. View the step-by-step solution
4. See which GSM8K examples were used as references

Example questions:
- "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. How many eggs does she sell daily?"
- "A shopkeeper bought 12 books for $5 each. He sold them for $8 each. What is his total profit?"
- "If 40% of a number is 20, what is the number?"

### Practice Mode
1. Click "Practice Mode" tab
2. Get a random problem from GSM8K test set
3. Try to solve it yourself
4. Click "Show Answer" to see the solution

## How to Modify

### Adjust Number of Retrieved Examples
In `src/generation.py`, change the `k` parameter:
```python
def answer_question(question: str, k: int = 4):  # Change 4 to your desired number
```

- Higher k = more context, but may be slower
- Lower k = faster, but less context

### Change LLM Model
In `src/generation.py`:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Try: gemini-1.5-pro for better quality
    temperature=0.2,            # 0 = deterministic, 1 = creative
```

### Use a Different Dataset
Modify `src/ingest.py` to load a different HuggingFace dataset or your own data.

## Troubleshooting

### "GOOGLE_API_KEY not set"
- Create a `.env` file with your Gemini API key
- Get one free at: https://aistudio.google.com/app/apikey

### "No module named 'datasets'"
```bash
pip install datasets
```

### Index build fails
- Check internet connection (needed to download GSM8K)
- Try clearing cache: `rm -rf ~/.cache/huggingface`

### Retrieval returns irrelevant examples
- Rebuild index: `python build_index.py`
- Try increasing k (retrieve more examples)
- Check if question is too different from GSM8K problems

## Dataset Attribution

This project uses the GSM8K dataset:
- **Paper**: "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)
- **Source**: OpenAI
- **License**: MIT
- **HuggingFace**: https://huggingface.co/datasets/openai/gsm8k

## License

This project is for educational purposes.

## Notes

Build the index from the notes pack.

Launch the Streamlit app.

Ask a question such as Find 25% of 80.

Show the retrieved note and source.

Repeat with questions from different topic families.

Open Practice Mode and show one sample question.

Current Scope
The current notes pack is aligned to these PSLE topic families:

Fractions and decimals

Percentage

Ratio and proportion

Rate / unitary reasoning

Measurement

Data handling

Future Improvements
Add grounded answer generation on top of retrieval

Improve citation formatting

Expand benchmark coverage

Add reranking or practice question generation extensions

text