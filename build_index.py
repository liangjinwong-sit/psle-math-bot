from src.ingest import get_all_documents
from src.retrieval import build_index

if __name__ == "__main__":
    """
    Build FAISS index from GSM8K dataset.
    
    GSM8K contains 8.5K grade school math word problems with solutions.
    - Training set: 7,473 examples
    - Test set: 1,319 examples
    
    We use the training set for RAG retrieval and keep test set for evaluation.
    """
    
    print("="*60)
    print("Building FAISS Index from GSM8K Dataset")
    print("="*60)
    
    # Load GSM8K training set for retrieval
    # You can limit this if you want a smaller index (faster but less coverage)
    documents = get_all_documents(
        use_train=True,      # Use full training set (7,473 examples)
        use_test=False,      # Keep test set separate for evaluation
        train_limit=None,    # None = use all training examples
    )
    
    print(f"\n📊 Building FAISS index with {len(documents)} documents...")
    build_index(documents)
    print("\n✅ Index build complete!")
    print("   - FAISS index saved to: index/psle_faiss/")
    print("   - Ready to run: streamlit run app.py")
