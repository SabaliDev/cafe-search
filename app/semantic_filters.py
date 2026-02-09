"""Semantic filtering using embeddings - actually leverages the 8GB of vectors."""

from __future__ import annotations

from typing import Sequence

from app.embeddings import embedding_service
from app.models import Job


# Concept expansions for semantic understanding
CONCEPT_EXPANSIONS = {
    # Seniority levels - be specific to avoid catching all managers
    "senior": "senior staff principal director VP head leadership experienced",
    "junior": "junior entry associate intern trainee graduate beginner entry-level",
    "lead": "lead leader tech lead team lead principal architect",
    
    # Industries (for company exclusion)
    "crypto": "cryptocurrency blockchain bitcoin ethereum web3 defi nft",
    "gambling": "gambling betting casino gaming lottery sportsbook",
    "tobacco": "tobacco cigarette smoking nicotine vaping",
    "oil": "oil gas petroleum fossil fuel drilling extraction",
}


def expand_concept(term: str) -> str:
    """Expand a simple term to a rich semantic concept."""
    term_lower = term.lower().strip()
    return CONCEPT_EXPANSIONS.get(term_lower, term_lower)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


def literal_match(job: Job, term: str) -> bool:
    """Check if job has literal match for term."""
    term_lower = term.lower()
    title_lower = job.title.lower()
    
    # Direct substring match
    if term_lower in title_lower:
        return True
    
    # Common variations
    variations = {
        "senior": ["sr.", "sr ", "staff", "principal"],
        "junior": ["jr.", "jr ", "associate", "entry"],
        "lead": ["leader", "leading"],
    }
    
    for var in variations.get(term_lower, []):
        if var in title_lower:
            return True
    
    return False


async def apply_semantic_exclusion(
    jobs: list[tuple[Job, float]],
    session,
    exclude_concepts: list[str],
    semantic_threshold: float = 0.36  # Catches directors (0.37+) but not coordinators (0.29-0.32)
) -> list[tuple[Job, float]]:
    """
    Hybrid exclusion: Literal matching + semantic similarity.
    
    Strategy:
    1. First, exclude literal matches (exact word in title)
    2. Then, use semantic similarity to catch related concepts
    
    Example for "exclude senior":
    - Literal: "Senior Manager" → excluded
    - Literal: "Sr. Engineer" → excluded  
    - Semantic: "Director" (similar to senior concept) → excluded if similarity > threshold
    - Keep: "Marketing Coordinator" (not similar to senior concept)
    
    This uses the 8GB of embeddings for what they're good at:
    - Understanding semantic similarity between concepts
    - Not just string matching
    """
    if not exclude_concepts:
        return jobs
    
    # Get expanded concepts and their embeddings
    expanded_concepts = []
    concept_embeddings = []
    
    for term in exclude_concepts:
        expanded = expand_concept(term)
        expanded_concepts.append(expanded)
        emb = await embedding_service.get_embedding(session, expanded)
        concept_embeddings.append(emb)
    
    filtered = []
    for job, score in jobs:
        # Check 1: Literal match (exact word in title)
        has_literal_match = any(literal_match(job, term) for term in exclude_concepts)
        
        if has_literal_match:
            # Definitely exclude
            continue
        
        # Check 2: Semantic similarity (using embeddings)
        job_embedding = job.embedding_explicit
        max_similarity = max(
            cosine_similarity(job_embedding, concept_emb)
            for concept_emb in concept_embeddings
        )
        
        # Only exclude if similarity is high enough
        # This catches "Director" when excluding "senior" but not "Coordinator"
        if max_similarity >= semantic_threshold:
            # Job is semantically similar to excluded concept
            continue
        
        # Keep the job
        filtered.append((job, score))
    
    return filtered


import asyncio
