# Research Paper Analysis Project

Research Paper Analysis Project is a semantic search system for machine learning research papers.
Instead of matching only exact keywords, it converts a research query into a
sentence embedding, searches an ArXiv paper index with FAISS, and then optionally
adds AI summaries and key phrases for each result.

This version has been customized from the base research-paper search idea with a
cleaner Streamlit interface, faster optional model loading, CSV export, similarity
filtering, safer cache checks, and command-line controls for building smaller demo
indexes.

## What The App Does

Enter a query such as:

```text
medical image segmentation with deep learning
```

The system will:

1. Embed the query with `sentence-transformers/all-MiniLM-L6-v2`.
2. Search a FAISS vector index built from ArXiv ML paper titles and abstracts.
3. Rank papers by cosine similarity.
4. Optionally summarize each abstract with a BART summarization model.
5. Optionally extract key phrases with KeyBERT.
6. Let the user download the visible results as a CSV file.

## Custom Features In This Version

- Renamed and redesigned Streamlit app: `Research Paper Analysis Project`.
- Fixed garbled UI/README encoding text.
- Added example-query buttons for faster demos.
- Added a minimum similarity filter.
- Added CSV export for search results.
- Added abstract word counts for each result.
- Lazy-loads summarizer and KeyBERT only when those options are enabled.
- Handles missing index/data files with clear setup commands inside the app.
- Added `--max-papers` and `--force` CLI options for data/index creation.
- Added cache-shape checks so stale embeddings or FAISS indexes are rebuilt.

## Tech Stack

| Layer | Tool |
|---|---|
| Dataset | `CShorten/ML-ArXiv-Papers` from Hugging Face |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | `faiss-cpu` with inner-product search over normalized vectors |
| Summarization | `sshleifer/distilbart-cnn-12-6` |
| Key Phrases | `keybert` |
| Data Handling | `pandas`, `numpy` |
| App UI | `streamlit` |

## Project Structure

```text
AI-Research-Paper-Intelligence-System-main/
|-- README.md
|-- requirements.txt
|-- data/
|   `-- README.md
|-- notebooks/
|   |-- 01_EDA_and_Embeddings.ipynb
|   `-- 02_Search_Engine.ipynb
`-- src/
    |-- app.py
    |-- build_index.py
    |-- data_prep.py
    `-- search_engine.py
```

## Setup

Create and activate a virtual environment if you want to keep dependencies
separate from other Python projects.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Build The Data And Index

For a quick demo, build a smaller index first:

```bash
python src/data_prep.py --max-papers 5000 --force
python src/build_index.py --max-papers 5000 --force
```

For the full project-sized index:

```bash
python src/data_prep.py --max-papers 50000 --force
python src/build_index.py --max-papers 50000 --force
```

The generated files are stored in `data/`:

- `cleaned_arxiv_papers.csv`
- `arxiv_embeddings.npy`
- `paper_faiss.index`

These files are not included in the repository because they are large and can be
regenerated from the source dataset.

## Run The App

```bash
streamlit run src/app.py
```

Open the local Streamlit URL, usually:

```text
http://localhost:8501
```

## Python Usage

```python
from src.search_engine import PaperSearchEngine

engine = PaperSearchEngine(load_summarizer=True, load_keybert=True)
results = engine.full_report(
    query="graph neural networks for drug discovery",
    k=5,
    include_summary=True,
    include_keywords=True,
    keyword_count=6,
)

for paper in results:
    print(paper["title"], paper["score"])
    print(paper.get("summary", ""))
    print(paper.get("keywords", []))
```

## Notes For Demo

- Building embeddings for 50,000 papers can take a while on CPU.
- Use `--max-papers 2000` or `--max-papers 5000` when recording or testing.
- The first app run is slower when summaries or keywords are enabled because
  transformer models need to load.
- If only semantic search is needed, disable summaries and key phrases in the
  sidebar for faster results.
