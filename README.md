# AI Placement and Resume Readiness System

This project is a Streamlit-based AI career readiness tool that analyzes a resume
against a target job description, estimates placement readiness, detects skill
gaps, creates a learning roadmap, and supports mock interview preparation.

The implementation is intentionally modular and local-first. It does not require
Gemini, OpenAI, FAISS, or a hosted notebook environment to run the core features.

## Problem Statement

Build an AI-powered placement preparation system using NLP and deep-learning
concepts that helps students evaluate resume quality, job-description alignment,
role readiness, and interview preparation gaps.

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
