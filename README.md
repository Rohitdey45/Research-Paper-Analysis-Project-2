# AI Placement and Resume Readiness System

> An AI-powered student placement preparation platform that analyzes resumes,
> checks ATS alignment, identifies role-specific skill gaps, estimates placement
> readiness, builds a personalized learning roadmap, and supports mock interview
> practice through a clean Streamlit interface.

**GitHub Repository:** https://github.com/Rohitdey45/Research-Paper-Analysis-Project-2

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-Resume%20Analysis-blueviolet)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Semantic%20Similarity-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Internship Details

| Field | Detail |
|---|---|
| Internship Title | AI/ML Internship |
| Organization | Coding Blocks School of Technology |
| Duration | June 2026 - July 2026 |
| Mentor / Supervisor | Aryesh Rai |
| Internship Type | Remote |

## Candidate Details

| Field | Detail |
|---|---|
| Name | Rohit Dey |
| Institution | Add your institution name |
| Course / Branch | Computer Science and Engineering |
| Roll No. / Registration No. | Add your roll number |
| Email | deyrohitd12@gmail.com |
| GitHub | https://github.com/Rohitdey45 |
| LinkedIn | Add your LinkedIn URL |

---

## Project Name

**AI Placement and Resume Readiness System**

## Project Description

Placement preparation is not just about having a resume. Students also need to
understand whether their resume matches a job description, which skills are
missing for a target role, how strong their overall profile is, and what they
should improve before applying.

This project brings those steps into one AI-assisted workflow. The system accepts
a resume and job description, extracts important resume signals with NLP, scores
ATS alignment, compares candidate skills with role requirements, estimates
placement readiness from academic and profile metrics, and generates a practical
daily/weekly/monthly improvement roadmap. It also includes a mock interview
practice section where students can answer role-specific questions and receive
structured feedback.

The implementation is modular, local-first, and explainable. It uses transparent
NLP and scoring rules with optional Sentence-Transformer similarity for
deep-learning based resume-job matching. The core app does not require a paid LLM
API key or a hosted notebook environment.

## Problem Statement

Given a student's resume, target job description, desired role, and profile
metrics, build an AI-powered system that evaluates resume quality, ATS alignment,
role readiness, placement probability, skill gaps, and interview preparation
needs, then provides actionable recommendations to improve placement readiness.

## Key Features

- Resume upload or paste support for PDF, TXT, and Markdown.
- ATS-style resume and job description matching.
- NLP-based skill extraction and keyword-gap analysis.
- Optional Sentence-Transformer semantic similarity for deep-learning based fit.
- Automatic fallback to TF-IDF/lexical scoring when transformer models are not installed.
- Role-based skill-gap analysis for AI Engineer, Data Scientist, MERN Developer,
  Backend Developer, Full Stack Developer, and Software Engineer.
- Placement readiness estimator using CGPA, DSA, aptitude, communication, projects,
  internships, certifications, hackathons, ATS score, and role skill readiness.
- Personalized daily, weekly, and monthly learning roadmap.
- Mock interview question bank and answer evaluator.
- JSON report export.
- One-command Windows launcher: `run_app.bat`.

## How This Version Is Different

- Uses Streamlit instead of Gradio.
- Uses separate Python modules instead of one large notebook.
- Runs without paid LLM/API keys.
- Keeps scoring transparent and explainable for project evaluation.
- Focuses on a practical student workflow: resume -> ATS -> skill gap -> readiness
  -> roadmap -> interview practice.

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| NLP | Regex parsing, TF-IDF, keyword extraction |
| Deep Learning | Sentence Transformers, optional |
| Data Handling | pandas, NumPy |
| ML Utilities | scikit-learn, optional fallback supported |
| PDF Parsing | pypdf |
| Language | Python |

## Project Structure

```text
Research-Paper-Analysis-Project-2/
|-- README.md
|-- requirements.txt
|-- run_app.bat
|-- data/
|   `-- README.md
`-- src/
    |-- app.py
    |-- career_coach.py
    `-- resume_analyzer.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

If you only want to test the Streamlit app quickly, install:

```bash
.venv\Scripts\python.exe -m pip install streamlit pandas pypdf
```

## Run The App

On Windows:

```bash
run_app.bat
```

Or run directly:

```bash
streamlit run src/app.py
```

Open:

```text
http://localhost:8501
```

## Application Tabs

| Tab | Purpose |
|---|---|
| Resume Input | Upload/paste resume, paste job description, run analysis |
| ATS Analysis | Shows ATS score, semantic fit, skill match, keyword match, and recommendations |
| Skill Gap | Compares detected resume skills with target role requirements |
| Placement Score | Estimates placement readiness from profile and resume signals |
| Roadmap | Generates daily, weekly, and monthly improvement plan |
| Mock Interview | Shows role-specific questions and evaluates answer quality |
| Export | Downloads the complete report as JSON |

## Scoring Logic

ATS score:

| Component | Weight |
|---|---:|
| Semantic similarity | 45% |
| Skill match | 30% |
| Keyword match | 15% |
| Resume section coverage | 10% |

Placement readiness combines:

- CGPA
- DSA score
- Aptitude score
- Communication score
- Project count
- Internship count
- Certification count
- Hackathon count
- Soft skills score
- ATS score
- Role skill readiness

## Notes

- The transformer model is optional. Keep it off for faster local demos.
- Real resumes may contain private information, so avoid committing them to GitHub.
- The placement readiness score is an explainable estimator for educational use,
  not a hiring guarantee.
