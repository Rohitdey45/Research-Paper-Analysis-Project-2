# AI Resume Analyzer using NLP and Deep Learning

This project analyzes a candidate resume against a target job description using
Natural Language Processing and deep-learning based semantic similarity. It
extracts resume text, detects important sections, identifies technical skills,
compares resume keywords with job requirements, and generates an overall match
score with improvement recommendations.

## Problem Statement

Build an AI Resume Analyzer using NLP and Deep Learning that helps candidates
understand how well their resume matches a job description and what changes can
improve their chances of shortlisting.

## Key Features

- Upload a resume as PDF, TXT, or Markdown.
- Paste a target job description for comparison.
- Extract skills from resume and job description.
- Compute skill match and missing skills.
- Extract and compare important job keywords.
- Detect resume sections such as Summary, Skills, Experience, Projects, and Education.
- Estimate experience years from resume text when available.
- Calculate semantic job-fit score using Sentence-Transformer embeddings.
- Fall back to TF-IDF similarity if the deep-learning model is unavailable.
- Generate practical resume improvement recommendations.
- Export the full analysis report as JSON.

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| NLP | TF-IDF, regex parsing, keyword extraction |
| Deep Learning | Sentence Transformers |
| ML Utilities | scikit-learn, NumPy |
| Data Handling | pandas |
| PDF Parsing | pypdf |

## Project Structure

```text
Research-Paper-Analysis-Project/
|-- README.md
|-- requirements.txt
|-- data/
|   `-- README.md
`-- src/
    |-- app.py
    `-- resume_analyzer.py
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run The App

```bash
streamlit run src/app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## How It Works

1. The user uploads a resume or pastes resume text.
2. The user pastes a target job description.
3. The app extracts resume signals such as skills, sections, contact details,
   experience phrases, and keywords.
4. A Sentence-Transformer model converts the resume and job description into
   dense vectors.
5. Cosine similarity between those vectors gives the semantic fit score.
6. The analyzer combines semantic fit, skill match, keyword match, and resume
   section coverage into one overall match score.
7. The app shows missing skills, missing keywords, and actionable suggestions.

## Scoring Logic

| Component | Weight |
|---|---:|
| Semantic similarity | 45% |
| Skill match | 30% |
| Keyword match | 15% |
| Resume section coverage | 10% |

## Notes

- The first deep-learning analysis can take longer because the transformer
  model needs to load.
- For faster testing, turn off the deep-learning model in the sidebar. The app
  will use TF-IDF similarity instead.
- Do not commit real resumes to GitHub because they may contain private data.
