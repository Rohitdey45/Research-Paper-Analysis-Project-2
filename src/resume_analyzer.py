"""
resume_analyzer.py
------------------
Core NLP logic for the AI Resume Analyzer app.

The module combines rule-based resume parsing, skill extraction, TF-IDF keyword
matching, and optional transformer embeddings for semantic job-fit scoring.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback for very small environments
    np = None

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - requirements.txt installs sklearn
    TfidfVectorizer = None
    cosine_similarity = None
    ENGLISH_STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SECTION_ALIASES = {
    "summary": ["summary", "profile", "objective", "about me"],
    "skills": ["skills", "technical skills", "core skills", "tools"],
    "experience": ["experience", "work experience", "employment", "internship"],
    "projects": ["projects", "academic projects", "personal projects"],
    "education": ["education", "academic background", "qualification"],
    "certifications": ["certifications", "certificates", "courses"],
    "achievements": ["achievements", "awards", "honors"],
}

SKILL_TAXONOMY = {
    "Programming": [
        "python",
        "java",
        "javascript",
        "typescript",
        "c",
        "c++",
        "c#",
        "sql",
        "r",
        "scala",
        "go",
        "php",
        "html",
        "css",
    ],
    "Data Science": [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "statistics",
        "data cleaning",
        "data visualization",
        "feature engineering",
        "eda",
    ],
    "Machine Learning": [
        "machine learning",
        "deep learning",
        "nlp",
        "natural language processing",
        "computer vision",
        "tensorflow",
        "keras",
        "pytorch",
        "scikit-learn",
        "sklearn",
        "transformers",
        "bert",
        "lstm",
        "cnn",
        "rnn",
        "xgboost",
        "classification",
        "regression",
        "clustering",
    ],
    "Backend": [
        "flask",
        "django",
        "fastapi",
        "node.js",
        "express",
        "rest api",
        "api",
        "mongodb",
        "mysql",
        "postgresql",
        "firebase",
    ],
    "Cloud And DevOps": [
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "git",
        "github",
        "linux",
        "ci/cd",
    ],
    "Analytics Tools": [
        "excel",
        "power bi",
        "tableau",
        "jupyter",
        "streamlit",
        "notebook",
    ],
    "Soft Skills": [
        "communication",
        "leadership",
        "teamwork",
        "problem solving",
        "critical thinking",
        "collaboration",
        "presentation",
    ],
}

ACTION_VERBS = {
    "built",
    "created",
    "developed",
    "designed",
    "implemented",
    "trained",
    "deployed",
    "optimized",
    "analyzed",
    "improved",
    "automated",
    "integrated",
    "managed",
    "led",
}


@dataclass
class ScoreBreakdown:
    semantic: float
    skills: float
    keywords: float
    sections: float
    overall: float


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving the original words."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def read_resume_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from a TXT or PDF resume upload."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if suffix == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to read PDF resumes.") from exc

        from io import BytesIO

        reader = PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return normalize_text("\n".join(pages))

    if suffix in {"txt", "md"}:
        return normalize_text(file_bytes.decode("utf-8", errors="ignore"))

    raise ValueError("Upload a PDF, TXT, or MD resume file.")


def _term_in_text(term: str, text_lower: str) -> bool:
    term_lower = term.lower()
    if re.search(r"[^a-z0-9\s]", term_lower):
        return term_lower in text_lower
    pattern = rf"(?<![a-z0-9]){re.escape(term_lower)}(?![a-z0-9])"
    return re.search(pattern, text_lower) is not None


def extract_skills(text: str) -> dict[str, list[str]]:
    """Return matched skills grouped by category."""
    text_lower = normalize_text(text).lower()
    matched: dict[str, list[str]] = {}

    for category, skills in SKILL_TAXONOMY.items():
        found = sorted({skill for skill in skills if _term_in_text(skill, text_lower)})
        if found:
            matched[category] = found

    return matched


def flatten_skills(skills_by_category: dict[str, list[str]]) -> set[str]:
    return {skill for skills in skills_by_category.values() for skill in skills}


def detect_sections(text: str) -> dict[str, bool]:
    """Detect common resume sections from heading-like text."""
    text_lower = text.lower()
    sections = {}
    for section, aliases in SECTION_ALIASES.items():
        sections[section] = any(
            re.search(rf"(^|\n|\r)\s*{re.escape(alias)}\s*[:\-]?", text_lower)
            or _term_in_text(alias, text_lower)
            for alias in aliases
        )
    return sections


def extract_contact_signals(text: str) -> dict[str, Any]:
    email_matches = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone_matches = re.findall(r"(?:\+?\d[\d\s().-]{8,}\d)", text)
    link_matches = re.findall(
        r"(?:https?://\S+|(?:linkedin|github)\.com/\S+)",
        text,
        flags=re.IGNORECASE,
    )
    return {
        "has_email": bool(email_matches),
        "has_phone": bool(phone_matches),
        "links": sorted(set(link_matches)),
    }


def estimate_experience_years(text: str) -> float | None:
    """Estimate experience from explicit phrases such as '2 years'."""
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", text, flags=re.IGNORECASE)
    if not matches:
        return None
    return max(float(match) for match in matches)


def extract_keywords(text: str, limit: int = 20) -> list[str]:
    """Extract important one- and two-word keywords using TF-IDF."""
    text = normalize_text(text)
    if not text:
        return []

    if TfidfVectorizer is not None:
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=max(limit * 3, 30),
            )
            matrix = vectorizer.fit_transform([text])
            scores = matrix.toarray()[0]
            terms = vectorizer.get_feature_names_out()
            ranked = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)
            return [term for term, score in ranked[:limit] if score > 0]
        except ValueError:
            pass

    words = re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", text.lower())
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return [word for word, _ in Counter(words).most_common(limit)]


def lexical_similarity(resume_text: str, job_description: str) -> float:
    """Small dependency-free similarity fallback based on shared content terms."""
    resume_terms = {
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", resume_text.lower())
        if word not in ENGLISH_STOP_WORDS
    }
    job_terms = {
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", job_description.lower())
        if word not in ENGLISH_STOP_WORDS
    }
    if not resume_terms or not job_terms:
        return 0.0
    return len(resume_terms.intersection(job_terms)) / len(job_terms)


def tfidf_similarity(resume_text: str, job_description: str) -> float:
    """Fallback semantic similarity when transformer embeddings are unavailable."""
    resume_text = normalize_text(resume_text)
    job_description = normalize_text(job_description)
    if not resume_text or not job_description:
        return 0.0

    if TfidfVectorizer is None or cosine_similarity is None:
        return lexical_similarity(resume_text, job_description)

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([resume_text, job_description])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except ValueError:
        return lexical_similarity(resume_text, job_description)


def transformer_similarity(model: Any, resume_text: str, job_description: str) -> float:
    """Compute deep-learning similarity using a sentence-transformer model."""
    resume_text = normalize_text(resume_text)
    job_description = normalize_text(job_description)
    if not resume_text or not job_description or model is None:
        return 0.0

    embeddings = model.encode(
        [resume_text[:4000], job_description[:4000]],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if np is not None:
        return float(np.dot(embeddings[0], embeddings[1]))
    return float(sum(left * right for left, right in zip(embeddings[0], embeddings[1])))


def section_score(sections: dict[str, bool]) -> float:
    required = ["summary", "skills", "experience", "projects", "education"]
    return sum(1 for section in required if sections.get(section)) / len(required)


def keyword_match_score(resume_text: str, job_keywords: list[str]) -> tuple[float, list[str], list[str]]:
    resume_lower = normalize_text(resume_text).lower()
    if not job_keywords:
        return 0.0, [], []

    matched = [keyword for keyword in job_keywords if keyword.lower() in resume_lower]
    missing = [keyword for keyword in job_keywords if keyword not in matched]
    return len(matched) / len(job_keywords), matched, missing


def build_recommendations(
    scores: ScoreBreakdown,
    missing_skills: list[str],
    missing_sections: list[str],
    matched_keywords: list[str],
    resume_text: str,
) -> list[str]:
    recommendations = []

    if missing_skills:
        recommendations.append(
            "Add or highlight these job-relevant skills: "
            + ", ".join(missing_skills[:8])
            + "."
        )

    if missing_sections:
        recommendations.append(
            "Improve resume structure by adding clear sections for "
            + ", ".join(missing_sections)
            + "."
        )

    if scores.semantic < 55:
        recommendations.append(
            "Tailor the summary and project bullets to mirror the job description more closely."
        )

    if scores.keywords < 45:
        recommendations.append(
            "Use more role-specific keywords from the job description in skills and experience bullets."
        )

    if not re.search(r"\d+%|\d+\+?|\b\d+x\b", resume_text, flags=re.IGNORECASE):
        recommendations.append(
            "Add measurable impact such as accuracy, time saved, users served, or percentage improvement."
        )

    action_count = sum(1 for verb in ACTION_VERBS if _term_in_text(verb, resume_text.lower()))
    if action_count < 4:
        recommendations.append(
            "Start more bullet points with strong action verbs like built, implemented, optimized, or deployed."
        )

    if not matched_keywords:
        recommendations.append(
            "Paste a detailed job description to get a stronger keyword and skill-gap analysis."
        )

    if not recommendations:
        recommendations.append(
            "Resume is well aligned. Keep the strongest projects near the top and maintain concise bullet points."
        )

    return recommendations


def analyze_resume(
    resume_text: str,
    job_description: str,
    role_title: str = "",
    embedding_model: Any | None = None,
) -> dict[str, Any]:
    """Run the complete resume analysis workflow."""
    resume_text = normalize_text(resume_text)
    job_description = normalize_text(job_description)
    if not resume_text:
        raise ValueError("Resume text is empty.")

    resume_skills_by_category = extract_skills(resume_text)
    job_skills_by_category = extract_skills(job_description)
    resume_skills = flatten_skills(resume_skills_by_category)
    job_skills = flatten_skills(job_skills_by_category)

    matched_skills = sorted(resume_skills.intersection(job_skills))
    missing_skills = sorted(job_skills.difference(resume_skills))
    skill_score = len(matched_skills) / len(job_skills) if job_skills else min(len(resume_skills) / 12, 1.0)

    job_keywords = extract_keywords(job_description, limit=25)
    keyword_score_value, matched_keywords, missing_keywords = keyword_match_score(resume_text, job_keywords)

    if embedding_model is not None and job_description:
        semantic_value = transformer_similarity(embedding_model, resume_text, job_description)
        similarity_method = "Sentence-Transformer"
    else:
        semantic_value = tfidf_similarity(resume_text, job_description)
        similarity_method = "TF-IDF fallback"

    sections = detect_sections(resume_text)
    section_value = section_score(sections)

    overall = (
        semantic_value * 0.45
        + skill_score * 0.30
        + keyword_score_value * 0.15
        + section_value * 0.10
    )
    scores = ScoreBreakdown(
        semantic=round(semantic_value * 100, 2),
        skills=round(skill_score * 100, 2),
        keywords=round(keyword_score_value * 100, 2),
        sections=round(section_value * 100, 2),
        overall=round(overall * 100, 2),
    )

    missing_sections = [
        section
        for section in ["summary", "skills", "experience", "projects", "education"]
        if not sections.get(section)
    ]

    contact = extract_contact_signals(resume_text)
    word_count = len(resume_text.split())

    return {
        "role_title": role_title,
        "scores": scores.__dict__,
        "similarity_method": similarity_method,
        "resume_stats": {
            "word_count": word_count,
            "estimated_experience_years": estimate_experience_years(resume_text),
            **contact,
        },
        "sections": sections,
        "resume_skills_by_category": resume_skills_by_category,
        "job_skills_by_category": job_skills_by_category,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords[:15],
        "recommendations": build_recommendations(
            scores=scores,
            missing_skills=missing_skills,
            missing_sections=missing_sections,
            matched_keywords=matched_keywords,
            resume_text=resume_text,
        ),
    }


def report_to_json(report: dict[str, Any]) -> bytes:
    return json.dumps(report, indent=2).encode("utf-8")
