# PSLE Math Study Bot

A Streamlit-based PSLE Math Study Bot that uses retrieval-augmented generation (RAG) over a team-curated notes pack for P5-P6 mathematics topics.

## Features
- Ask a PSLE-style math question
- Retrieve relevant notes from the indexed knowledge base
- Show supporting source chunks
- Practice mode with question and answer reveal
- Local FAISS index using HuggingFace embeddings

## Project Structure
```text
psle-math-bot/
├─ app.py
├─ build_index.py
├─ requirements.txt
├─ data/
│  ├─ notes/
│  │  ├─ topic_fractions_decimals.md
│  │  ├─ topic_percentage.md
│  │  ├─ topic_ratio_proportion.md
│  │  ├─ topic_rate.md
│  │  ├─ topic_measurement.md
│  │  └─ topic_data_handling.md
│  └─ benchmark/
│     ├─ benchmark_questions.md
│     └─ evaluation_checklist.md
├─ index/
│  └─ psle_faiss/
└─ src/
   ├─ ingest.py
   ├─ retrieval.py
   ├─ generation.py
   ├─ ui.py
   └─ practice.py
Installation
Create and activate a Python environment.

Install the dependencies:

bash
pip install -r requirements.txt
Build the Index
After adding or editing notes in data/notes/, rebuild the FAISS index:

bash
python build_index.py
If successful, you should see a message saying the FAISS index was built and saved.

Run the App
Start the Streamlit app with:

bash
streamlit run app.py
If streamlit is not recognized, use:

bash
python -m streamlit run app.py
Then open the local URL shown in the terminal, usually:

text
http://localhost:8501
How to Use
Open the Ask a Question tab.

Type a PSLE-style math question.

Click the search button to retrieve relevant notes.

Review the retrieved notes and source labels.

Open Practice Mode to generate a practice question and reveal the answer or full working.

Notes
The current version uses local retrieval over Markdown notes in data/notes/.

The FAISS index must be rebuilt whenever the notes are changed.

Markdown is used as the primary authoring format because it is cleaner and easier to chunk than PDF extraction.

The current MVP focuses on grounded retrieval, citation support, and clean demo flow.

Evaluation
Benchmark and evaluation files are stored in:

text
data/benchmark/
Suggested evaluation process:

Run each benchmark question through the app.

Check whether the retrieved notes match the expected topic.

Check whether the retrieved notes support the expected method and answer.

Record results in evaluation_checklist.md.

Demo Flow
A simple demo sequence:

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