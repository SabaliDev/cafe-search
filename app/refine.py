from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from openai import AsyncOpenAI

from app.intent import classify_intent, IntentResult
from app.schemas import SearchFilters
from app.config import settings
from app.logger import log_event

client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None


# OpenAI pricing per 1M tokens (as of 2025)
PRICING = {
    "gpt-4o-mini": {
        "input": 0.15,   # $0.15 per 1M input tokens
        "output": 0.60,  # $0.60 per 1M output tokens
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    "text-embedding-3-small": {
        "input": 0.02,   # $0.02 per 1M tokens
        "output": 0.0,
    },
    "text-embedding-3-large": {
        "input": 0.13,
        "output": 0.0,
    },
}


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    
    @classmethod
    def from_response(cls, prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini"):
        pricing = PRICING.get(model, PRICING["gpt-4o-mini"])
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            input_cost_usd=round(input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            total_cost_usd=round(input_cost + output_cost, 6),
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "input_cost_usd": self.input_cost_usd,
            "output_cost_usd": self.output_cost_usd,
            "total_cost_usd": self.total_cost_usd,
        }

SYSTEM_PROMPT = """You are a job search assistant. Your task is to update job search filters based on a user's refinement query and conversation history.

Current Filters: {current_filters}
Conversation History: {history}

The user's query may add, remove, or modify filters. 
Possible filters:
- remote (boolean)
- org_types (list of strings: startup, enterprise, non-profit, government, b-corp)
- salary_min (integer)
- salary_max (integer)
- location (string)
- country (string, e.g., "US", "CA", "UK")
- experience_level (string: senior, junior)
- mission_keywords (list of strings)
- company_industry (list of strings)
- inferred_titles (list of strings)
- title_keywords (list of strings) - for matching against job titles specifically
- description_keywords (list of strings) - for matching against job descriptions specifically
- exclude_title_keywords (list of strings) - keywords to EXCLUDE from titles (e.g., "senior", "lead", "principal")
- exclude_company_names (list of strings) - company names/industries to EXCLUDE (e.g., "crypto", "gambling")

IMPORTANT: Only output the filters that are CHANGING. Do not repeat unchanged filters.
- To ADD a filter, include it in the output
- To REMOVE a filter, set its value to null
- If a filter is not mentioned, do NOT include it (it will be preserved automatically)

NEGATIVE CONSTRAINTS (Exclusions):
- When user says "exclude X", "no X", "without X", or "not X", add to exclude_title_keywords or exclude_company_names
- Examples: "exclude senior" → add "senior" to exclude_title_keywords
- Examples: "no crypto companies" → add "crypto" to exclude_company_names
- Examples: "without lead roles" → add "lead" to exclude_title_keywords

Special commands:
- If the user wants to see "something different" or "else", set "clear_previous" to true.

EXTRACTION: Extract specialization_term if the user is refining by a domain/skill/area.
- Examples: "tax" → "tax", "something in audit" → "audit", "with payroll experience" → "payroll"
- This is a single term or short phrase representing the domain/skill, NOT the full job title.
- If no specialization is mentioned, set to null.

Output a JSON object with:
1. "filter_changes": Only the filters that are being added, modified, or removed (null = remove).
2. "changes": A human-readable description of what changed.
3. "clear_previous": boolean.
4. "new_job_type": If the user is searching for a NEW type of job (not refining), set this to the job type string. Otherwise null.
5. "specialization_term": The extracted domain/skill term or null if none.
6. "is_role_specialization": boolean - true if the term is clearly a role/skill (e.g., "tax accountant"), false if ambiguous (e.g., "tax" could be company type).

Examples:
User: "make it remote"
Output: {{"filter_changes": {{"remote": true}}, "changes": {{"remote": true}}, "clear_previous": false, "new_job_type": null, "specialization_term": null, "is_role_specialization": false}}

User: "actually, show me data science jobs"
Output: {{"filter_changes": {{}}, "changes": {{"job_type": "data science"}}, "clear_previous": true, "new_job_type": "data science jobs", "specialization_term": null, "is_role_specialization": false}}

User: "with tax experience"
Output: {{"filter_changes": {{}}, "changes": {{"specialization": "tax"}}, "clear_previous": false, "new_job_type": null, "specialization_term": "tax", "is_role_specialization": false}}

User: "tax accountant roles"
Output: {{"filter_changes": {{"inferred_titles": ["Tax Accountant"]}}, "changes": {{"specialization": "tax accountant"}}, "clear_previous": false, "new_job_type": null, "specialization_term": "tax accountant", "is_role_specialization": true}}

User: "exclude senior"
Output: {{"filter_changes": {{"exclude_title_keywords": ["senior"]}}, "changes": {{"excluded": "senior"}}, "clear_previous": false, "new_job_type": null, "specialization_term": null, "is_role_specialization": false}}

User: "no crypto or gambling companies"
Output: {{"filter_changes": {{"exclude_company_names": ["crypto", "gambling"]}}, "changes": {{"excluded_companies": ["crypto", "gambling"]}}, "clear_previous": false, "new_job_type": null, "specialization_term": null, "is_role_specialization": false}}
"""

async def apply_refinement(
    query: str,
    current_filters: dict[str, Any],
    persistent_filters: list[str],
    history: list[str] = None
) -> tuple[SearchFilters, dict[str, Any], bool, IntentResult, str | None, str | None, bool, TokenUsage | None]:
    """
    Apply refinement to search filters.
    
    Returns:
        - new_filters: The merged filter set
        - changes: Description of what changed
        - clear_previous: Whether to exclude previous results
        - intent_result: Intent classification result
        - new_job_type: If user wants a completely new job type, otherwise None
        - specialization_term: Extracted domain/skill term from LLM, or None
        - is_role_specialization: Whether the term is clearly a role/skill
        - token_usage: Token usage statistics from LLM call
    """
    history_str = "\n".join(history[-5:]) if history else "None"
    
    prompt = f"User Query: {query}"
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(
                    current_filters=json.dumps(current_filters),
                    history=history_str
                )},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        filter_changes = result.get("filter_changes", result.get("filters", {}))
        # Normalize legacy/alias keys from LLM outputs
        if "exclude_title_keywords_semantic" in filter_changes and "exclude_title_keywords" not in filter_changes:
            filter_changes["exclude_title_keywords"] = filter_changes.pop("exclude_title_keywords_semantic")
        changes = result.get("changes", {})
        clear_previous = result.get("clear_previous", False)
        new_job_type = result.get("new_job_type")
        
        # Extract specialization info from LLM response
        llm_specialization = result.get("specialization_term")
        is_role_specialization = result.get("is_role_specialization", False)
        
        # Extract token usage from response
        token_usage = None
        model_name = "gpt-4o-mini"
        if response.usage:
            token_usage = TokenUsage.from_response(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                model=model_name,
            )
            # Log token usage with cost
            log_event(
                "llm_token_usage",
                {
                    "model": model_name,
                    "operation": "refinement",
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens,
                    "input_cost_usd": token_usage.input_cost_usd,
                    "output_cost_usd": token_usage.output_cost_usd,
                    "total_cost_usd": token_usage.total_cost_usd,
                    "query": query,
                },
            )

        # Normalize location synonyms (e.g., USA -> US)
        if "location" in filter_changes and isinstance(filter_changes["location"], str):
            loc = filter_changes["location"].strip().lower()
            if loc in {"usa", "u.s.a", "united states", "united states of america"}:
                filter_changes.pop("location", None)
                filter_changes["country"] = "US"

        # Normalize exclude keywords (e.g., "management" -> manager variants)
        if "exclude_title_keywords" in filter_changes:
            normalized = []
            for kw in filter_changes["exclude_title_keywords"]:
                k = kw.lower().strip()
                if k == "management":
                    normalized.extend(["manager", "management", "mgr", "lead", "director"])
                else:
                    normalized.append(k)
            filter_changes["exclude_title_keywords"] = list(dict.fromkeys(normalized))

        # Enforce filter schema: drop unknown keys
        allowed_keys = set(SearchFilters.model_fields.keys())
        filter_changes = {k: v for k, v in filter_changes.items() if k in allowed_keys}

        # Merge: start with current filters, apply changes
        merged = {**current_filters}
        for key, value in filter_changes.items():
            if value is None:
                # Remove filter
                merged.pop(key, None)
            else:
                # Add or update filter
                merged[key] = value
        
        new_filters = SearchFilters(**merged)
        intent_result = classify_intent(query)
        
        return new_filters, changes, clear_previous, intent_result, new_job_type, llm_specialization, is_role_specialization, token_usage
    except Exception as e:
        # Fallback
        intent_result = classify_intent(query)
        log_event(
            "llm_token_usage",
            {
                "model": "gpt-4o-mini",
                "operation": "refinement",
                "error": str(e),
                "query": query,
            },
        )
        return SearchFilters(**current_filters), {"error": str(e)}, False, intent_result, None, None, False, None
