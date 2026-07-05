"""
app.py
------
Streamlit front-end for Research Paper Analysis Project.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from search_engine import PaperSearchEngine


st.set_page_config(
    page_title="Research Paper Analysis Project",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_engine(load_summarizer: bool, load_keybert: bool) -> PaperSearchEngine:
    return PaperSearchEngine(
        load_summarizer=load_summarizer,
        load_keybert=load_keybert,
    )


def to_download_csv(results: list[dict]) -> bytes:
    rows = []
    for result in results:
        rows.append(
            {
                "rank": result["rank"],
                "similarity": round(result["score"], 4),
                "title": result["title"],
                "summary": result.get("summary", ""),
                "keywords": ", ".join(keyword for keyword, _ in result.get("keywords", [])),
                "abstract": result["abstract"],
            }
        )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


if "query" not in st.session_state:
    st.session_state.query = ""


st.title("Research Paper Analysis Project")
st.caption(
    "Semantic paper discovery for machine learning research: search by meaning, "
    "scan summaries, compare similarity scores, and export the strongest matches."
)

with st.sidebar:
    st.header("Search controls")
    top_k = st.slider("Results to retrieve", min_value=1, max_value=15, value=5)
    min_score = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
    )
    show_summary = st.checkbox("Generate summaries", value=True)
    show_keywords = st.checkbox("Extract key phrases", value=True)
    keyword_count = st.slider("Key phrases per paper", min_value=3, max_value=12, value=6)

    st.divider()
    st.subheader("Pipeline")
    st.markdown(
        "Query -> sentence embedding -> FAISS semantic search -> optional "
        "summary and key phrase extraction."
    )

st.subheader("Research query")
examples = [
    "transformer models for time series forecasting",
    "graph neural networks for drug discovery",
    "federated learning privacy techniques",
    "medical image segmentation with deep learning",
]

example_cols = st.columns(len(examples))
for col, example in zip(example_cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state.query = example

query = st.text_input(
    "Search research papers",
    key="query",
    placeholder="Type a topic, method, problem, or research question",
    label_visibility="collapsed",
)

search_clicked = st.button("Search papers", type="primary")

if search_clicked and query.strip():
    try:
        engine = get_engine(
            load_summarizer=show_summary,
            load_keybert=show_keywords,
        )
    except FileNotFoundError as exc:
        st.error("The search index is not ready yet.")
        st.write("Run these commands once from the project folder:")
        st.code(
            "python src/data_prep.py\n"
            "python src/build_index.py\n"
            "streamlit run src/app.py",
            language="bash",
        )
        st.caption(str(exc))
        st.stop()

    with st.spinner("Searching the paper index..."):
        results = engine.full_report(
            query=query,
            k=top_k,
            include_summary=show_summary,
            include_keywords=show_keywords,
            keyword_count=keyword_count,
        )

    filtered_results = [result for result in results if result["score"] >= min_score]

    if not filtered_results:
        st.warning("No papers passed the selected similarity threshold.")
        st.stop()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Papers shown", len(filtered_results))
    metric_cols[1].metric("Top similarity", f"{filtered_results[0]['score']:.3f}")
    metric_cols[2].metric("Threshold", f"{min_score:.2f}")

    st.download_button(
        "Download results as CSV",
        data=to_download_csv(filtered_results),
        file_name="research_paper_analysis_results.csv",
        mime="text/csv",
    )

    for result in filtered_results:
        with st.container(border=True):
            header_cols = st.columns([6, 1])
            with header_cols[0]:
                st.subheader(f"{result['rank']}. {result['title']}")
                st.caption(f"Abstract length: {result['abstract_word_count']} words")
            with header_cols[1]:
                st.metric("Similarity", f"{result['score']:.3f}")

            st.progress(min(max(result["score"], 0.0), 1.0))

            if show_summary and result.get("summary"):
                st.markdown("**AI summary**")
                st.write(result["summary"])

            if show_keywords and result.get("keywords"):
                st.markdown("**Key phrases**")
                st.write(
                    "  ".join(
                        f"`{keyword}` ({score:.2f})"
                        for keyword, score in result["keywords"]
                    )
                )

            with st.expander("Read full abstract"):
                st.write(result["abstract"])

elif search_clicked:
    st.warning("Enter a search query first.")
else:
    st.info("Enter a query or choose an example to start exploring the paper database.")
