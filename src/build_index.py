"""
build_index.py
---------------
Converts every cleaned paper into a 384-dimensional sentence embedding
(using all-MiniLM-L6-v2) and stores them in a FAISS index for fast
similarity search. Both the embeddings (.npy) and the index (.index)
are cached to disk so this expensive step only needs to run once.

Run directly:
    python src/build_index.py
"""

import argparse
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from data_prep import MAX_PAPERS, get_cleaned_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "arxiv_embeddings.npy")
INDEX_PATH = os.path.join(DATA_DIR, "paper_faiss.index")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 32


def load_model() -> SentenceTransformer:
    print(f"Loading sentence-transformer model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def build_embeddings(df, model: SentenceTransformer, force: bool = False) -> np.ndarray:
    """Encode every paper's text into a vector, caching the result to disk."""
    if not force and os.path.exists(EMBEDDINGS_PATH):
        print("Loading cached embeddings...")
        embeddings = np.load(EMBEDDINGS_PATH)
        if embeddings.shape[0] == len(df) and embeddings.shape[1] == EMBEDDING_DIM:
            return embeddings.astype(np.float32)
        print("Cached embeddings do not match the current dataset; rebuilding.")

    print(f"Encoding {len(df):,} papers... (this can take a while on CPU)")
    embeddings = model.encode(
        df["paper_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype(np.float32)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray, force: bool = False) -> faiss.Index:
    """
    Build (or load) a FAISS index using Inner Product search on L2-normalized
    vectors, which is mathematically equivalent to Cosine Similarity but
    much faster to query at scale.
    """
    if not force and os.path.exists(INDEX_PATH):
        print("Loading cached FAISS index...")
        index = faiss.read_index(INDEX_PATH)
        if index.ntotal == embeddings.shape[0] and index.d == embeddings.shape[1]:
            return index
        print("Cached FAISS index does not match the current embeddings; rebuilding.")

    print("Building new FAISS index...")
    # Work on a copy: FAISS normalizes in-place and we don't want to mutate
    # the original embeddings array that other steps may still rely on.
    normed = embeddings.copy()
    faiss.normalize_L2(normed)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(normed)

    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")
    return index


def build_all(force: bool = False, max_papers: int = MAX_PAPERS):
    """Full pipeline: clean data -> embeddings -> FAISS index."""
    df = get_cleaned_data(force_reload=force, max_papers=max_papers)
    model = load_model()
    embeddings = build_embeddings(df, model, force=force)
    index = build_faiss_index(embeddings, force=force)
    return df, model, embeddings, index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build embeddings and a FAISS index.")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=MAX_PAPERS,
        help="Maximum number of papers to keep when preparing data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cleaned data, embeddings, and the FAISS index.",
    )
    args = parser.parse_args()
    build_all(force=args.force, max_papers=args.max_papers)
