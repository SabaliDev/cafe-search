from __future__ import annotations

from typing import Sequence


ROLE_ANCHORS: dict[str, list[str]] = {
    "data science": ["Data Scientist", "ML Engineer", "Machine Learning Engineer", "Data Analyst", "Analytics Engineer"],
    "software engineer": ["Software Engineer", "Software Developer", "Backend Engineer", "Backend Developer", "Full Stack Engineer", "Full Stack Developer", "Frontend Engineer", "Frontend Developer"],
    "product manager": ["Product Manager", "Product Owner", "Technical Product Manager"],
    "designer": ["Product Designer", "UX Designer", "UI Designer", "UX Researcher", "Graphic Designer"],
    "data analyst": ["Data Analyst", "Business Analyst", "Analytics Engineer"],
}

# Expanded keywords for semantic matching
ROLE_KEYWORDS: dict[str, list[str]] = {
    "software engineer": [
        "software", "developer", "engineer", "programmer", "coder",
        "backend", "frontend", "fullstack", "full-stack", "web",
        "java", "python", "go", "golang", "rust", "javascript", "typescript",
        "node", "react", "angular", "vue",
    ],
    "data science": [
        "data", "ml", "machine learning", "ai", "model", 
        "analytics", "statistics", "research",
    ],
    "product manager": [
        "product", "pm", "roadmap", "strategy",
    ],
    "designer": [
        "design", "ux", "ui", "visual", "creative", "graphic",
    ],
    "data analyst": [
        "analysis", "reporting", "dashboard", "sql", "excel",
    ],
}


def extract_role_anchor(query: str) -> list[str] | None:
    """Extract role anchor from query using keyword matching.
    
    First tries exact key match, then falls back to keyword scoring.
    """
    normalized = query.lower()
    
    # First: exact key match
    for key, roles in ROLE_ANCHORS.items():
        if key in normalized:
            return roles
    
    # Second: keyword-based scoring for variations like "Backend Developer"
    scores: dict[str, int] = {}
    for role, keywords in ROLE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in normalized)
        if score > 0:
            scores[role] = score
    
    # Return best match if it has at least 2 keyword hits or strong single hit
    if scores:
        best_role = max(scores, key=scores.get)
        if scores[best_role] >= 2 or (scores[best_role] == 1 and len(normalized.split()) <= 3):
            return ROLE_ANCHORS[best_role]
    
    return None
