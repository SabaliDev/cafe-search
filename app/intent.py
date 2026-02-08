import re
from dataclasses import dataclass


EXPLICIT_KEYWORDS = {
    "engineer",
    "scientist",
    "analyst",
    "developer",
    "designer",
    "product",
    "data",
    "ml",
    "ai",
    "machine learning",
    "analytics",
    "backend",
    "frontend",
    "fullstack",
    "devops",
    "security",
    "cloud",
    "mobile",
}

COMPANY_KEYWORDS = {
    "culture",
    "mentorship",
    "mission",
    "values",
    "social good",
    "impact",
    "non-profit",
    "nonprofit",
    "government",
    "public sector",
    "b-corp",
    "benefit corporation",
    "sustainability",
    "diversity",
    "inclusion",
    "equity",
}

INFERRED_KEYWORDS = {
    "climate",
    "fintech",
    "healthcare",
    "biotech",
    "education",
    "edtech",
    "proptech",
    "agtech",
    "energy",
    "transportation",
    "logistics",
    "space",
    "defense",
    "cybersecurity",
}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: list[str]
    breakdown: dict[str, float]


def _score_keywords(query: str, keywords: set[str]) -> float:
    score = 0.0
    for keyword in keywords:
        if keyword in query:
            score += 1.0
    return score


def _extract_entities(query: str) -> list[str]:
    phrases = re.findall(r"\b[a-zA-Z][a-zA-Z+\- ]{2,40}\b", query)
    cleaned = []
    for phrase in phrases:
        phrase = phrase.strip().lower()
        if phrase in {"jobs", "job", "role", "roles", "position", "positions"}:
            continue
        if phrase not in cleaned:
            cleaned.append(phrase)
    return cleaned[:10]


def classify_intent(query: str) -> IntentResult:
    normalized = query.lower().strip()
    explicit_score = _score_keywords(normalized, EXPLICIT_KEYWORDS)
    company_score = _score_keywords(normalized, COMPANY_KEYWORDS)
    inferred_score = _score_keywords(normalized, INFERRED_KEYWORDS)

    breakdown = {
        "explicit": explicit_score,
        "company": company_score,
        "inferred": inferred_score,
    }

    max_score = max(breakdown.values())
    if max_score == 0:
        return IntentResult(
            intent="inferred",
            confidence=0.2,
            entities=_extract_entities(normalized),
            breakdown=breakdown,
        )

    intent = max(breakdown, key=breakdown.get)
    confidence = min(0.9, 0.4 + 0.15 * max_score)
    return IntentResult(
        intent=intent,
        confidence=confidence,
        entities=_extract_entities(normalized),
        breakdown=breakdown,
    )
