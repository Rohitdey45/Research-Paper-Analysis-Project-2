"""
app.py
------
Streamlit front-end for AI Resume Analyzer using NLP and Deep Learning.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from resume_analyzer import (
    DEFAULT_MODEL_NAME,
    analyze_resume,
    read_resume_file,
    report_to_json,
)


st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def skills_frame(report: dict, key: str) -> pd.DataFrame:
    rows = []
    for category, skills in report.get(key, {}).items():
        for skill in skills:
            rows.append({"Category": category, "Skill": skill})
    return pd.DataFrame(rows)


if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""


st.title("AI Resume Analyzer")
st.caption(
    "Analyze a resume against a job description using NLP keyword extraction, "
    "skill matching, section detection, and transformer-based semantic similarity."
)

with st.sidebar:
    st.header("Analysis settings")
    role_title = st.text_input("Target role", placeholder="Data Scientist / ML Engineer")
    use_deep_model = st.checkbox("Use deep learning semantic model", value=True)
    st.caption("When enabled, Sentence-Transformer embeddings are used for job-fit scoring.")

    st.divider()
    st.subheader("Expected resume sections")
    st.write("Summary, Skills, Experience, Projects, Education")

st.subheader("Resume input")
upload = st.file_uploader("Upload resume", type=["pdf", "txt", "md"])
if upload is not None:
    try:
        st.session_state.resume_text = read_resume_file(upload.getvalue(), upload.name)
        st.success(f"Loaded resume text from {upload.name}")
    except Exception as exc:
        st.error(str(exc))

resume_text = st.text_area(
    "Resume text",
    value=st.session_state.resume_text,
    height=260,
    placeholder="Upload a resume or paste resume text here.",
)
st.session_state.resume_text = resume_text

st.subheader("Job description")
job_description = st.text_area(
    "Paste job description",
    height=220,
    placeholder=(
        "Paste the target job description here so the analyzer can compute "
        "skill gap, keyword match, and semantic similarity."
    ),
)

analyze_clicked = st.button("Analyze resume", type="primary")

if analyze_clicked:
    if not resume_text.strip():
        st.warning("Upload or paste resume text first.")
        st.stop()

    embedding_model = None
    if use_deep_model and job_description.strip():
        try:
            with st.spinner("Loading transformer model..."):
                embedding_model = load_embedding_model()
        except Exception as exc:
            st.warning(
                "Transformer model could not be loaded. Falling back to TF-IDF similarity."
            )
            st.caption(str(exc))

    with st.spinner("Analyzing resume..."):
        report = analyze_resume(
            resume_text=resume_text,
            job_description=job_description,
            role_title=role_title,
            embedding_model=embedding_model,
        )

    scores = report["scores"]
    metric_cols = st.columns(5)
    metric_cols[0].metric("Overall match", f"{scores['overall']:.1f}%")
    metric_cols[1].metric("Semantic fit", f"{scores['semantic']:.1f}%")
    metric_cols[2].metric("Skill match", f"{scores['skills']:.1f}%")
    metric_cols[3].metric("Keyword match", f"{scores['keywords']:.1f}%")
    metric_cols[4].metric("Section score", f"{scores['sections']:.1f}%")
    st.caption(f"Similarity method: {report['similarity_method']}")

    st.progress(min(max(scores["overall"] / 100, 0.0), 1.0))

    overview_cols = st.columns(3)
    overview_cols[0].metric("Resume words", report["resume_stats"]["word_count"])
    overview_cols[1].metric(
        "Experience",
        (
            f"{report['resume_stats']['estimated_experience_years']} years"
            if report["resume_stats"]["estimated_experience_years"] is not None
            else "Not detected"
        ),
    )
    overview_cols[2].metric(
        "Contact info",
        "Present"
        if report["resume_stats"]["has_email"] and report["resume_stats"]["has_phone"]
        else "Incomplete",
    )

    tab_summary, tab_skills, tab_keywords, tab_sections, tab_export = st.tabs(
        ["Recommendations", "Skills", "Keywords", "Sections", "Export"]
    )

    with tab_summary:
        st.subheader("Improvement recommendations")
        for index, recommendation in enumerate(report["recommendations"], start=1):
            st.write(f"{index}. {recommendation}")

    with tab_skills:
        skill_cols = st.columns(2)
        with skill_cols[0]:
            st.subheader("Resume skills")
            resume_skill_df = skills_frame(report, "resume_skills_by_category")
            if resume_skill_df.empty:
                st.info("No known skills detected.")
            else:
                st.dataframe(resume_skill_df, use_container_width=True, hide_index=True)

        with skill_cols[1]:
            st.subheader("Job-required skills")
            job_skill_df = skills_frame(report, "job_skills_by_category")
            if job_skill_df.empty:
                st.info("Paste a detailed job description to detect required skills.")
            else:
                st.dataframe(job_skill_df, use_container_width=True, hide_index=True)

        st.subheader("Skill gap")
        gap_cols = st.columns(2)
        gap_cols[0].write("Matched skills")
        gap_cols[0].write(", ".join(report["matched_skills"]) or "No matched skills yet.")
        gap_cols[1].write("Missing skills")
        gap_cols[1].write(", ".join(report["missing_skills"]) or "No missing skills detected.")

    with tab_keywords:
        keyword_cols = st.columns(2)
        keyword_cols[0].subheader("Matched job keywords")
        keyword_cols[0].write(", ".join(report["matched_keywords"]) or "No keyword matches yet.")
        keyword_cols[1].subheader("Missing job keywords")
        keyword_cols[1].write(", ".join(report["missing_keywords"]) or "No missing keywords detected.")

    with tab_sections:
        section_rows = [
            {"Section": section.title(), "Detected": "Yes" if detected else "No"}
            for section, detected in report["sections"].items()
        ]
        st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)

    with tab_export:
        st.download_button(
            "Download analysis report",
            data=report_to_json(report),
            file_name="resume_analysis_report.json",
            mime="application/json",
        )
        st.json(report)

else:
    st.info("Upload a resume, paste a job description, and click Analyze resume.")
