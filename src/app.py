"""
app.py
------
Streamlit front-end for AI Placement and Resume Readiness System.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from career_coach import (
    build_learning_roadmap,
    estimate_placement_readiness,
    evaluate_mock_answer,
    get_mock_questions,
    get_role_names,
    readiness_gauge_label,
    role_skill_gap,
)
from resume_analyzer import (
    DEFAULT_MODEL_NAME,
    analyze_resume,
    flatten_skills,
    read_resume_file,
    report_to_json,
)


st.set_page_config(
    page_title="AI Placement Readiness System",
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


def show_score_cards(scores: dict) -> None:
    metric_cols = st.columns(5)
    metric_cols[0].metric("Overall ATS", f"{scores['overall']:.1f}%")
    metric_cols[1].metric("Semantic fit", f"{scores['semantic']:.1f}%")
    metric_cols[2].metric("Skill match", f"{scores['skills']:.1f}%")
    metric_cols[3].metric("Keyword match", f"{scores['keywords']:.1f}%")
    metric_cols[4].metric("Section score", f"{scores['sections']:.1f}%")
    st.progress(min(max(scores["overall"] / 100, 0.0), 1.0))


if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "analysis_report" not in st.session_state:
    st.session_state.analysis_report = None
if "role_gap" not in st.session_state:
    st.session_state.role_gap = None
if "placement_report" not in st.session_state:
    st.session_state.placement_report = None


st.title("AI Placement and Resume Readiness System")
st.caption(
    "A focused Streamlit project for resume analysis, ATS matching, skill-gap "
    "tracking, placement readiness scoring, roadmap planning, and mock interview prep."
)

with st.sidebar:
    st.header("Candidate setup")
    target_role = st.selectbox("Target role", get_role_names(), index=1)
    use_deep_model = st.checkbox("Use deep learning semantic model", value=False)
    st.caption("If unavailable, the app automatically uses TF-IDF/lexical similarity.")

    st.divider()
    st.subheader("Profile metrics")
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
    dsa = st.slider("DSA score", 0, 100, 65)
    aptitude = st.slider("Aptitude score", 0, 100, 70)
    communication = st.slider("Communication score", 0, 100, 70)
    projects = st.slider("Projects", 0, 8, 2)
    internships = st.slider("Internships", 0, 5, 1)
    certifications = st.slider("Certifications", 0, 8, 2)
    hackathons = st.slider("Hackathons", 0, 6, 1)
    soft_skills = st.slider("Soft skills score", 0, 100, 70)

tab_input, tab_ats, tab_gap, tab_placement, tab_roadmap, tab_interview, tab_export = st.tabs(
    [
        "Resume Input",
        "ATS Analysis",
        "Skill Gap",
        "Placement Score",
        "Roadmap",
        "Mock Interview",
        "Export",
    ]
)

with tab_input:
    input_cols = st.columns([1, 1])

    with input_cols[0]:
        st.subheader("Resume")
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
            height=340,
            placeholder="Upload a resume or paste resume text here.",
        )
        st.session_state.resume_text = resume_text

    with input_cols[1]:
        st.subheader("Job description")
        job_description = st.text_area(
            "Paste target job description",
            height=340,
            placeholder=(
                "Paste a job description for ATS matching, keyword gap, and semantic fit."
            ),
        )

        analyze_clicked = st.button("Run complete analysis", type="primary", use_container_width=True)

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
                st.warning("Transformer model unavailable. Using TF-IDF fallback.")
                st.caption(str(exc))

        with st.spinner("Analyzing resume and profile..."):
            report = analyze_resume(
                resume_text=resume_text,
                job_description=job_description,
                role_title=target_role,
                embedding_model=embedding_model,
            )
            resume_skills = flatten_skills(report["resume_skills_by_category"])
            gap = role_skill_gap(resume_skills, target_role)
            placement = estimate_placement_readiness(
                profile={
                    "cgpa": cgpa,
                    "dsa": dsa,
                    "aptitude": aptitude,
                    "communication": communication,
                    "projects": projects,
                    "internships": internships,
                    "certifications": certifications,
                    "hackathons": hackathons,
                    "soft_skills": soft_skills,
                },
                ats_score=report["scores"]["overall"],
                skill_score=gap["role_readiness"],
            )

        st.session_state.analysis_report = report
        st.session_state.role_gap = gap
        st.session_state.placement_report = placement
        st.success("Analysis completed. Open the other tabs to review results.")

with tab_ats:
    report = st.session_state.analysis_report
    if not report:
        st.info("Run complete analysis from the Resume Input tab first.")
    else:
        st.subheader("ATS and resume analysis")
        show_score_cards(report["scores"])
        st.caption(f"Similarity method: {report['similarity_method']}")

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

        st.subheader("Recommendations")
        for index, recommendation in enumerate(report["recommendations"], start=1):
            st.write(f"{index}. {recommendation}")

        keyword_cols = st.columns(2)
        keyword_cols[0].write("Matched job keywords")
        keyword_cols[0].write(", ".join(report["matched_keywords"]) or "No keyword matches yet.")
        keyword_cols[1].write("Missing job keywords")
        keyword_cols[1].write(", ".join(report["missing_keywords"]) or "No missing keywords detected.")

with tab_gap:
    report = st.session_state.analysis_report
    gap = st.session_state.role_gap
    if not report or not gap:
        st.info("Run complete analysis from the Resume Input tab first.")
    else:
        st.subheader(f"Skill gap for {gap['role']}")
        gap_cols = st.columns(3)
        gap_cols[0].metric("Role readiness", f"{gap['role_readiness']:.1f}%")
        gap_cols[1].metric("Matched skills", len(gap["matched_role_skills"]))
        gap_cols[2].metric("Missing skills", len(gap["missing_role_skills"]))
        st.progress(min(max(gap["role_readiness"] / 100, 0.0), 1.0))

        skill_cols = st.columns(2)
        with skill_cols[0]:
            st.subheader("Detected resume skills")
            resume_skill_df = skills_frame(report, "resume_skills_by_category")
            if resume_skill_df.empty:
                st.info("No known skills detected.")
            else:
                st.dataframe(resume_skill_df, use_container_width=True, hide_index=True)

        with skill_cols[1]:
            st.subheader("Required role skills")
            st.write(", ".join(gap["required_skills"]))
            st.write("Missing skills")
            st.write(", ".join(gap["missing_role_skills"]) or "No missing role skills detected.")

with tab_placement:
    placement = st.session_state.placement_report
    if not placement:
        st.info("Run complete analysis from the Resume Input tab first.")
    else:
        st.subheader("Placement readiness estimator")
        cols = st.columns(3)
        cols[0].metric("Placement readiness", f"{placement['placement_probability']:.1f}%")
        cols[1].metric("Level", placement["readiness_level"])
        cols[2].metric("Label", readiness_gauge_label(placement["placement_probability"]))
        st.progress(min(max(placement["placement_probability"] / 100, 0.0), 1.0))

        st.write("Priority improvement areas")
        st.write(", ".join(placement["improvement_areas"]))

        normalized_df = pd.DataFrame(
            [
                {"Signal": key.replace("_", " ").title(), "Score": round(value * 100, 2)}
                for key, value in placement["normalized_inputs"].items()
            ]
        )
        st.dataframe(normalized_df, use_container_width=True, hide_index=True)

with tab_roadmap:
    gap = st.session_state.role_gap
    placement = st.session_state.placement_report
    if not gap or not placement:
        st.info("Run complete analysis from the Resume Input tab first.")
    else:
        roadmap = build_learning_roadmap(
            target_role,
            gap["missing_role_skills"],
            placement["improvement_areas"],
        )
        st.subheader("Personalized learning roadmap")
        roadmap_cols = st.columns(3)
        with roadmap_cols[0]:
            st.write("Daily plan")
            for item in roadmap["daily_plan"]:
                st.write(f"- {item}")
        with roadmap_cols[1]:
            st.write("Weekly plan")
            for item in roadmap["weekly_plan"]:
                st.write(f"- {item}")
        with roadmap_cols[2]:
            st.write("Monthly plan")
            for item in roadmap["monthly_plan"]:
                st.write(f"- {item}")

        st.write("Focus skills")
        st.write(", ".join(roadmap["focus_skills"]))

with tab_interview:
    st.subheader("Mock interview preparation")
    questions = get_mock_questions(target_role)
    selected_question = st.selectbox("Choose a question", questions)
    answer = st.text_area("Your answer", height=180)
    if st.button("Evaluate answer"):
        if not answer.strip():
            st.warning("Write an answer first.")
        else:
            evaluation = evaluate_mock_answer(answer, target_role)
            st.metric("Answer score", f"{evaluation['answer_score']:.1f}%")
            st.write("Covered role terms")
            st.write(", ".join(evaluation["covered_terms"]) or "No role terms detected.")
            st.write("Feedback")
            for item in evaluation["feedback"]:
                st.write(f"- {item}")

with tab_export:
    report = st.session_state.analysis_report
    gap = st.session_state.role_gap
    placement = st.session_state.placement_report
    if not report:
        st.info("Run complete analysis from the Resume Input tab first.")
    else:
        full_report = {
            "resume_analysis": report,
            "role_gap": gap,
            "placement_readiness": placement,
        }
        st.download_button(
            "Download full report",
            data=report_to_json(full_report),
            file_name="placement_readiness_report.json",
            mime="application/json",
        )
        st.json(full_report)
