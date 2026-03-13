import argparse
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
    
    parser = argparse.ArgumentParser(description="Build FAISS index from GSM8K dataset")
    parser.add_argument(
        "--chunk-mode",
        type=str,
        default="full",
        choices=["full", "step", "hybrid"],
        help="Chunking mode: full (legacy), step, or hybrid",
    )
    parser.add_argument(
        "--step-window-size",
        type=int,
        default=3,
        help="Number of solution steps per chunk window when chunk-mode is step/hybrid",
    )
    parser.add_argument(
        "--step-overlap",
        type=int,
        default=1,
        help="Number of overlapping steps between adjacent windows",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include GSM8K test split in index build (off by default)",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional limit for number of train rows to ingest",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Optional limit for number of test rows to ingest",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Building FAISS Index from GSM8K Dataset")
    print("=" * 60)
    print(
        f"Chunk config: mode={args.chunk_mode}, "
        f"step_window_size={args.step_window_size}, step_overlap={args.step_overlap}"
    )

    # Load GSM8K data for retrieval index build.
    documents = get_all_documents(
        use_train=True,
        use_test=args.include_test,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        chunk_mode=args.chunk_mode,
        step_window_size=args.step_window_size,
        step_overlap=args.step_overlap,
        include_full_in_hybrid=True,
    )

    print(f"\nBuilding FAISS index with {len(documents)} documents...")
    build_index(documents)
    print("\nIndex build complete!")
    print("   - FAISS index saved to: index/psle_faiss/")
    print("   - Ready to run: streamlit run app.py")
