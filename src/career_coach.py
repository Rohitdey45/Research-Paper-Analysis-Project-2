"""
career_coach.py
---------------
Career-readiness helpers for the Streamlit placement preparation app.

This module intentionally uses transparent scoring rules instead of external LLM
APIs so the project can run locally and be explained clearly during evaluation.
"""

from __future__ import annotations

import math
import re
from typing import Any


ROLE_PROFILES = {
    "AI Engineer": {
        "skills": [
            "python",
            "machine learning",
            "deep learning",
            "nlp",
            "tensorflow",
            "pytorch",
            "transformers",
            "api",
            "docker",
            "git",
        ],
        "project": "Build an end-to-end model deployment project with a simple API and dashboard.",
    },
    "Data Scientist": {
        "skills": [
            "python",
            "sql",
            "statistics",
            "pandas",
            "numpy",
            "data visualization",
            "machine learning",
            "classification",
            "regression",
            "power bi",
        ],
        "project": "Create a business analytics case study with EDA, model comparison, and insights.",
    },
    "MERN Developer": {
        "skills": [
            "javascript",
            "typescript",
            "html",
            "css",
            "mongodb",
            "express",
            "node.js",
            "api",
            "git",
        ],
        "project": "Build a full-stack application with auth, CRUD, filtering, and deployment notes.",
    },
    "Backend Developer": {
        "skills": [
            "python",
            "java",
            "sql",
            "flask",
            "django",
            "fastapi",
            "api",
            "postgresql",
            "docker",
            "git",
        ],
        "project": "Build a REST API with authentication, database models, tests, and documentation.",
    },
    "Full Stack Developer": {
        "skills": [
            "javascript",
            "typescript",
            "html",
            "css",
            "python",
            "node.js",
            "api",
            "mongodb",
            "postgresql",
            "git",
        ],
        "project": "Create a production-style web app with frontend, backend, database, and deployment.",
    },
    "Software Engineer": {
        "skills": [
            "python",
            "java",
            "c++",
            "sql",
            "git",
            "problem solving",
            "data structures",
            "algorithms",
            "api",
        ],
        "project": "Prepare a clean software project with algorithms, tests, and a strong README.",
    },
}


INTERVIEW_BANK = {
    "AI Engineer": [
        "Explain the difference between machine learning and deep learning with one project example.",
        "How would you reduce overfitting in a neural network?",
        "What steps would you follow to deploy an NLP model as an API?",
        "How do transformer models understand context better than traditional models?",
    ],
    "Data Scientist": [
        "How do you handle missing values and outliers in a dataset?",
        "Explain precision, recall, and F1-score with an example.",
        "How would you explain a model's prediction to a non-technical stakeholder?",
        "What is the difference between correlation and causation?",
    ],
    "MERN Developer": [
        "Explain how frontend and backend communicate in a MERN application.",
        "How would you design authentication for a full-stack web app?",
        "What are common performance issues in React applications?",
        "How do you structure REST APIs for maintainability?",
    ],
    "Backend Developer": [
        "How would you design a scalable REST API?",
        "Explain database indexing and when it helps.",
        "How do you handle authentication and authorization in backend systems?",
        "What logging and error-handling practices do you follow?",
    ],
    "Full Stack Developer": [
        "How do you decide which logic belongs on frontend vs backend?",
        "Explain your approach to designing a responsive dashboard.",
        "How would you debug a slow full-stack application?",
        "What steps do you follow before deploying a web app?",
    ],
    "Software Engineer": [
        "Explain time complexity using a problem you solved.",
        "How do you write maintainable code in a team project?",
        "Describe a bug you fixed and how you found the root cause.",
        "How would you approach a new problem in an interview?",
    ],
}


def get_role_names() -> list[str]:
    return sorted(ROLE_PROFILES)


def role_skill_gap(resume_skills: set[str], role: str) -> dict[str, Any]:
    required = set(ROLE_PROFILES.get(role, {}).get("skills", []))
    matched = sorted(resume_skills.intersection(required))
    missing = sorted(required.difference(resume_skills))
    readiness = round((len(matched) / len(required)) * 100, 2) if required else 0.0
    return {
        "role": role,
        "required_skills": sorted(required),
        "matched_role_skills": matched,
        "missing_role_skills": missing,
        "role_readiness": readiness,
    }


def estimate_placement_readiness(profile: dict[str, float], ats_score: float, skill_score: float) -> dict[str, Any]:
    """Estimate placement readiness with an explainable weighted score."""
    weights = {
        "cgpa": 0.10,
        "dsa": 0.18,
        "aptitude": 0.12,
        "communication": 0.12,
        "projects": 0.12,
        "internships": 0.10,
        "certifications": 0.06,
        "hackathons": 0.05,
        "soft_skills": 0.05,
        "ats": 0.06,
        "skills": 0.04,
    }

    normalized = {
        "cgpa": min(profile.get("cgpa", 0) / 10, 1),
        "dsa": profile.get("dsa", 0) / 100,
        "aptitude": profile.get("aptitude", 0) / 100,
        "communication": profile.get("communication", 0) / 100,
        "projects": min(profile.get("projects", 0) / 5, 1),
        "internships": min(profile.get("internships", 0) / 3, 1),
        "certifications": min(profile.get("certifications", 0) / 5, 1),
        "hackathons": min(profile.get("hackathons", 0) / 4, 1),
        "soft_skills": profile.get("soft_skills", 0) / 100,
        "ats": ats_score / 100,
        "skills": skill_score / 100,
    }

    weighted = sum(normalized[key] * weights[key] for key in weights)
    probability = round(weighted * 100, 2)
    level = "High" if probability >= 75 else "Medium" if probability >= 50 else "Needs Work"

    weakest = sorted(normalized.items(), key=lambda item: item[1])[:3]
    improvements = [key.replace("_", " ").title() for key, _ in weakest]

    return {
        "placement_probability": probability,
        "readiness_level": level,
        "improvement_areas": improvements,
        "normalized_inputs": normalized,
    }


def build_learning_roadmap(role: str, missing_skills: list[str], improvement_areas: list[str]) -> dict[str, Any]:
    priority_skills = missing_skills[:6]
    role_project = ROLE_PROFILES.get(role, {}).get("project", "Build one polished role-specific project.")

    if not priority_skills:
        priority_skills = ["advanced project polish", "interview revision", "deployment", "documentation"]

    return {
        "daily_plan": [
            f"Revise one core concept from {priority_skills[0]} and write short notes.",
            "Solve 3 role-relevant practice problems or implementation tasks.",
            "Update one resume bullet with measurable impact.",
        ],
        "weekly_plan": [
            f"Complete a mini project task around {priority_skills[0]}.",
            "Practice one mock interview and review weak answers.",
            "Push clean code and improve README/project screenshots.",
        ],
        "monthly_plan": [
            role_project,
            "Apply to 15-20 targeted roles with tailored resumes.",
            "Track interview feedback and revise weak areas every week.",
        ],
        "focus_skills": priority_skills,
        "profile_focus": improvement_areas,
    }


def get_mock_questions(role: str, count: int = 4) -> list[str]:
    questions = INTERVIEW_BANK.get(role, INTERVIEW_BANK["Software Engineer"])
    return questions[:count]


def evaluate_mock_answer(answer: str, role: str) -> dict[str, Any]:
    text = answer.strip()
    words = re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", text.lower())
    required_terms = set(ROLE_PROFILES.get(role, ROLE_PROFILES["Software Engineer"])["skills"])
    covered_terms = sorted(term for term in required_terms if term in text.lower())

    length_score = min(len(words) / 80, 1) * 35
    keyword_score = min(len(covered_terms) / 4, 1) * 35
    structure_score = 15 if any(marker in text.lower() for marker in ["first", "then", "finally", "because"]) else 5
    example_score = 15 if re.search(r"\b(project|built|implemented|created|deployed|improved)\b", text.lower()) else 5
    score = round(length_score + keyword_score + structure_score + example_score, 2)

    feedback = []
    if len(words) < 50:
        feedback.append("Make the answer more detailed with a clear problem, action, and result.")
    if len(covered_terms) < 2:
        feedback.append("Use more role-specific technical terms from your projects.")
    if example_score < 10:
        feedback.append("Add one real project example to make the answer stronger.")
    if not feedback:
        feedback.append("Good answer structure. Add metrics if possible to make it more convincing.")

    return {
        "answer_score": min(score, 100),
        "covered_terms": covered_terms,
        "feedback": feedback,
    }


def readiness_gauge_label(score: float) -> str:
    if math.isnan(score):
        return "Not calculated"
    if score >= 75:
        return "Strong"
    if score >= 50:
        return "Moderate"
    return "Needs work"
