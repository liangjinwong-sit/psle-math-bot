# Quick Setup Guide

# Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up Gemini API Key
1. Get a free API key at: https://aistudio.google.com/app/apikey
2. Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Step 3: Build Index and Run
```bash
# Build FAISS index from GSM8K dataset (takes 5-10 minutes)
python build_index.py

# Run the app
streamlit run app.py
```

That's it! Open http://localhost:8501 in your browser.

---

## 📚 What This Does

Your bot uses **RAG (Retrieval-Augmented Generation)**:

1. **Loads GSM8K Dataset** - 7,500+ math word problems with solutions
2. **Creates Vector Index** - FAISS stores embeddings for fast search
3. **Retrieves Similar Problems** - Finds examples similar to student's question
4. **Generates Solution** - Gemini LLM creates step-by-step answer using examples

---

## 💡 Example Questions to Try

- "Janet's ducks lay 16 eggs per day. She eats three for breakfast. How many eggs does she sell daily if she bakes muffins with 4?"
- "A shopkeeper bought 12 books for $5 each and sold them for $8 each. What is his total profit?"
- "If 40% of a number is 20, what is the number?"

---

## 🔧 Troubleshooting

**"GOOGLE_API_KEY not set"**
- Create `.env` file with your API key

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**"Index not found"**
```bash
python build_index.py
```

---

See [README.md](README.md) for full documentation.
