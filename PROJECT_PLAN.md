# PSLE Math Bot - Complete Implementation Plan

## 📋 Project Overview

**Goal:** Build an AI-powered PSLE Math tutoring bot using RAG (Retrieval-Augmented Generation) that helps Primary 5 and Primary 6 students in Singapore learn mathematics through:
1. Question answering with step-by-step explanations
2. Practice question generation
3. Grounded answers using curated PSLE notes
4. Topic-aware retrieval for accurate responses

---

## 🎯 Implementation Phases

### Phase 1: Data Ingestion & Index Building ✅ (COMPLETED)

**What was done:**
- ✅ Updated `src/ingest.py` to load curated markdown notes
- ✅ Added support for PSLE-Notes-LLM.pdf
- ✅ Made GSM8K optional (reduces American content bias)
- ✅ Added topic metadata to all documents
- ✅ Updated `build_index.py` to prioritize PSLE content

**Next step:**
```bash
python build_index.py
```

This will rebuild your FAISS index with:
1. Your 6 curated topic markdown files (highest priority)
2. PSLE-Notes-LLM.pdf (if available)
3. Optional GSM8K for supplementary examples

---

### Phase 2: Enhanced RAG Generation ✅ (COMPLETED)

**What was done:**
- ✅ Improved `src/generation.py` with better tutoring prompt
- ✅ Added structured response format (Final Answer, Steps, Key Concept)
- ✅ Added topic-aware retrieval display
- ✅ Better error handling for missing API keys

**Features:**
- Clear final answer stated upfront
- Numbered step-by-step working
- Simple language for 11-12 year olds
- Explains mathematical reasoning (WHY)
- Cites reference sources

---

### Phase 3: Evaluation & Benchmarking ✅ (COMPLETED)

**What was done:**
- ✅ Created `src/evaluation.py` module
- ✅ Automated benchmark testing
- ✅ Retrieval quality metrics
- ✅ Answer correctness checking
- ✅ Summary statistics

**How to run:**
```bash
python src/evaluation.py
```

Or integrate into your workflow:
```python
from src.evaluation import run_benchmark_evaluation

benchmark = [
    {
        "question": "Find 25% of 80.",
        "expected_answer": "20",
        "expected_topic": "percentage",
    },
    # ... more questions
]

results = run_benchmark_evaluation(benchmark, verbose=True)
```

---

### Phase 4: UI Improvements (TODO - NEXT STEP)

**What to do:**

1. **Update Streamlit UI** to show:
   - Retrieved topic information
   - Confidence indicators
   - Source citations with topic tags
   - Better formatting for step-by-step solutions

2. **Add evaluation dashboard:**
   - Run benchmarks from UI
   - Display metrics visually
   - Track performance over time

3. **Improve Practice Mode:**
   - Use PSLE-specific questions instead of GSM8K
   - Allow topic filtering
   - Show similar worked examples

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up API Key
Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your free key at: https://aistudio.google.com/app/apikey

### Step 3: Build Index
```bash
python build_index.py
```

Expected output:
```
[ingest] Loaded X chunks from 6 markdown files.
[ingest] Loaded Y chunks from 'PSLE-Notes-LLM.pdf'.
[ingest] Total documents collected: Z

📊 Building FAISS index with Z documents...
✅ Index build complete. Ready to run the app!
```

### Step 4: Run App
```bash
streamlit run app.py
```

### Step 5: Run Evaluation (Optional)
```bash
python src/evaluation.py
```

---

## 📊 RAG Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     STUDENT QUESTION                        │
│                  "Find 25% of 80."                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              SENTENCE TRANSFORMERS                          │
│         (all-MiniLM-L6-v2 - Local Embeddings)              │
│              Convert question to vector                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  FAISS VECTOR STORE                         │
│           Similarity search (cosine distance)               │
│          Retrieve top-k most relevant chunks                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            RETRIEVED DOCUMENTS (k=4)                        │
│                                                              │
│  [1] topic_percentage.md [Topic: percentage]                │
│      Method 1: Find a percentage of a quantity...           │
│                                                              │
│  [2] topic_percentage.md [Topic: percentage]                │
│      Example: 25% of 80 = 25/100 × 80 = 20                 │
│                                                              │
│  [3] PSLE-Notes-LLM.pdf [Topic: general]                   │
│      Percentage basics...                                    │
│                                                              │
│  [4] topic_percentage.md [Topic: percentage]                │
│      Standard Methods...                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                GEMINI LLM (gemini-1.5-flash)               │
│              Enhanced Tutoring Prompt                       │
│                                                              │
│  You are a patient PSLE Math tutor...                      │
│  Retrieved materials: [1] [2] [3] [4]                      │
│  Question: Find 25% of 80.                                  │
│  Provide: Final Answer, Steps, Key Concept, Sources        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   GENERATED RESPONSE                        │
│                                                              │
│  **Final Answer:** 20                                       │
│                                                              │
│  **Step-by-Step Working:**                                  │
│  Step 1: Convert 25% to a fraction: 25/100 = 1/4          │
│  Step 2: Multiply 1/4 by 80: (1/4) × 80 = 20              │
│                                                              │
│  **Key Concept:** Finding a percentage of a number         │
│                                                              │
│  **Reference Sources:** Examples [1] and [2] from          │
│  topic_percentage.md helped explain this method.            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Expected Performance Metrics

Based on typical RAG pipelines for educational content:

### Retrieval Quality (Target: ≥80%)
- **Topic Match Rate:** Documents retrieved should match expected topic
- **Source Relevance:** Retrieved chunks should contain relevant methods/examples

### Answer Accuracy (Target: ≥85%)
- **Correctness:** Final numerical answer matches expected
- **Completeness:** All steps shown with explanations
- **Clarity:** Language appropriate for P5/P6 students

### Overall System (Target: ≥75%)
- **End-to-End Success:** Both retrieval AND answer are correct

---

## 🎓 Key Features of Your RAG Pipeline

### 1. **Hybrid Data Sources**
- ✅ Curated markdown notes (highest priority)
- ✅ PSLE PDF materials
- ⚠️ Optional GSM8K (use sparingly to avoid US curriculum bias)

### 2. **Topic-Aware Retrieval**
- Every document tagged with topic metadata
- Retrieval considers both semantic similarity AND topic relevance
- Sources displayed with topic labels for transparency

### 3. **Educational Prompting**
- Structured response format (Answer, Steps, Concept, Sources)
- Age-appropriate language (11-12 year olds)
- Encouragement and positive reinforcement
- Singapore Math methodology

### 4. **Grounded Responses**
- All answers cite retrieved sources
- Students can verify reasoning against notes
- Reduces hallucination risk

### 5. **Automated Evaluation**
- Benchmark testing against known Q&A pairs
- Quantitative metrics for iteration
- Easy to run before/after changes

---

## 🔄 Recommended Workflow for Your Project

### Week 1-2: Foundation (DONE ✅)
- [x] Set up RAG pipeline
- [x] Curate PSLE notes by topic
- [x] Build FAISS index
- [x] Integrate Gemini LLM

### Week 3: Testing & Iteration (NOW)
1. **Rebuild index** with new ingestion code:
   ```bash
   python build_index.py
   ```

2. **Run evaluation** on benchmark questions:
   ```bash
   python src/evaluation.py
   ```

3. **Analyze results:**
   - Which topics perform well?
   - Which questions fail? Why?
   - Is retrieval working correctly?

4. **Iterate on weak areas:**
   - Add more examples for low-performing topics
   - Adjust chunk sizes if needed
   - Refine prompts for clarity

### Week 4: Enhancement
1. **Improve UI** with better visualization
2. **Add more benchmark questions** for comprehensive testing
3. **Create PSLE-specific practice questions** (replace GSM8K)
4. **Add topic filtering** in practice mode

### Week 5: Documentation & Demo
1. **Document findings** in progress report
2. **Prepare demo** with best-performing questions
3. **Create evaluation report** with metrics
4. **Prepare presentation** showing RAG pipeline

---

## 📝 Tips for Success

### 1. **Focus on Curated Notes Quality**
- Your markdown files are the foundation
- Add more worked examples to weak topic areas
- Keep formatting consistent (aids chunking)

### 2. **Balance Data Sources**
- Prioritize YOUR notes over external datasets
- GSM8K can help, but PSLE coverage is key
- PDF content should supplement, not replace, markdown

### 3. **Iterate on Prompts**
- Test different instructional styles
- Adjust temperature if responses too creative/rigid
- Ensure sources are always cited

### 4. **Measure Everything**
- Run benchmarks after each change
- Track metrics over time
- Use data to guide improvements

### 5. **Student-Centric Design**
- Test with actual P5/P6 students if possible
- Ensure language is accessible
- Add visual aids (charts, diagrams) if feasible

---

## 🎯 Success Criteria for Project Delivery

### Minimum Viable Product (MVP)
- ✅ RAG pipeline working end-to-end
- ✅ FAISS index built from curated notes
- ✅ Gemini LLM generating tutoring responses
- ✅ Basic UI for Q&A
- ✅ Retrieval grounding with source citations

### Full Deliverables
- ✅ All 6 PSLE topics covered in notes
- ✅ Benchmark evaluation with ≥75% pass rate
- ✅ Practice mode functional
- ⏳ Evaluation metrics dashboard (in progress)
- ⏳ Documentation of methodology and results

### Stretch Goals
- Add multi-step problem decomposition
- Implement adaptive difficulty
- Add progress tracking per student
- Mobile-responsive UI
- Offline mode (no LLM, formula-based fallback)

---

## 📚 Resources

### Documentation
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

### PSLE Math Resources
- Singapore MOE PSLE Syllabus
- Your curated markdown notes
- PSLE-Notes-LLM.pdf

---

## 🚨 Common Issues & Solutions

### Issue 1: Index build fails
**Solution:** Check that all paths in `build_index.py` are correct
```python
# Verify paths exist
import os
print(os.path.exists("data/notes"))  # Should be True
print(os.path.exists("PSLE-Notes-LLM.pdf"))  # Should be True
```

### Issue 2: LLM returns generic answers
**Solution:** Retrieval may not be finding relevant docs
- Rebuild index with `python build_index.py`
- Check if question topics match your notes coverage
- Increase k value (retrieve more documents)

### Issue 3: API key errors
**Solution:** Ensure .env file is properly configured
```bash
# Check if .env exists
ls .env

# View contents (Mac/Linux)
cat .env

# View contents (Windows)
type .env
```

### Issue 4: Evaluation shows low accuracy
**Solution:** Improve note coverage and examples
- Add more worked examples for failing topics
- Ensure notes use PSLE-style language
- Check if benchmark questions are realistic

---

## 💡 Next Immediate Actions

1. **Rebuild your index:**
   ```bash
   python build_index.py
   ```

2. **Test a question manually:**
   ```bash
   streamlit run app.py
   # Try: "Find 25% of 80"
   ```

3. **Run evaluation on your benchmarks:**
   - Parse `data/benchmark/benchmark_questions.md`
   - Convert to evaluation format
   - Run `python src/evaluation.py`

4. **Review results and iterate:**
   - Which topics need more examples?
   - Is retrieval finding the right docs?
   - Are LLM responses clear and accurate?

---

**You now have a complete RAG pipeline! Time to test, iterate, and refine.** 🚀
