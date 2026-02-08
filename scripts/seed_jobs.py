import argparse
import asyncio
import json
import binascii
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal
from app.models import Job, OrgType


def _parse_line(line: str) -> dict[str, Any]:
    line = line.strip()
    if not line:
        return {}
    if line.startswith("{"):
        return json.loads(line)
    try:
        decoded = binascii.unhexlify(line).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        raise ValueError("Unrecognized jobs.jsonl line format")


def _get_embedding(data: dict[str, Any], key: str) -> list[float] | None:
    candidates = [
        key,
        f"{key}_vector",
        key.replace("embedding_", ""),
        key.replace("embedding_", "") + "_vector",
        key.replace("embedding_", "") + "_embedding",
        key.replace("embedding_", "") + "_embeddings",
    ]
    containers = [
        data,
        data.get("embeddings") or {},
        data.get("v7_processed_job_data") or {},
        data.get("v5_processed_job_data") or {},
    ]
    for container in containers:
        for candidate in candidates:
            if candidate in container:
                return container[candidate]
    return None


def _coerce_org_type(data: dict[str, Any]) -> OrgType | None:
    info = data.get("job_information") or {}
    v5 = data.get("v5_processed_job_data") or {}
    v7 = data.get("v7_processed_job_data") or {}
    
    val = data.get("org_type") or info.get("org_type") or v5.get("org_type") or v7.get("org_type")
    
    # Also check industry/sector/tagline
    all_text = str(v5.get("company_sector_and_industry") or "") + \
               str(v5.get("company_tagline") or "") + \
               str(v5.get("company_activities") or "") + \
               str(v7.get("company_sector_and_industry") or "") + \
               str(v7.get("company_tagline") or "") + \
               str(v7.get("company_activities") or "")
    all_text = all_text.lower()
    
    if "non-profit" in all_text or "nonprofit" in all_text:
        return OrgType.non_profit
    if "startup" in all_text:
        return OrgType.startup

    if not val:
        return None
    val = str(val).lower().strip()
    if val == "nonprofit":
        val = "non-profit"
    if val in OrgType._value2member_map_:
        return OrgType(val)
    return None


def _extract_location_from_title(title: str) -> str | None:
    """Extract location from title in format 'Job Title (City, ST, Country, Zip)'."""
    if not title:
        return None
    match = re.search(r'\(([^)]+)\)\s*$', title)
    if match:
        loc = match.group(1)
        # Heuristic: if it looks like a location (contains comma)
        if ',' in loc:
            return loc
    return None


def _extract_job_info(data: dict[str, Any]) -> dict[str, Any]:
    info = data.get("job_information") or {}
    v5 = data.get("v5_processed_job_data") or {}
    v7 = data.get("v7_processed_job_data") or {}
    v5_company = data.get("v5_processed_company_data") or {}
    v7_company = data.get("v7_processed_company_data") or {}
    
    title = data.get("title") or info.get("title") or info.get("job_title_raw")
    description = data.get("description") or info.get("description") or ""
    
    # Fix 4: Extract company name from v5_processed_company_data
    company_name = (
        data.get("company_name") or 
        info.get("company_name") or 
        data.get("company") or
        v7_company.get("name") or
        v5_company.get("name")
    )
    
    # Fix 3: Extract location from title if not directly available
    location = data.get("location") or info.get("location")
    if not location:
        location = _extract_location_from_title(title or "")
    
    remote = data.get("remote")
    if remote is None:
        remote = v5.get("is_remote") or v7.get("is_remote")
        
    if remote is None:
        search_text = f"{title or ''} {location or ''}".lower()
        remote = "remote" in search_text
    if remote is None:
        remote = False
    org_type = _coerce_org_type(data)
    salary_min = data.get("salary_min") or info.get("salary_min")
    salary_max = data.get("salary_max") or info.get("salary_max")
    posted_at = data.get("posted_at") or info.get("posted_at")
    if posted_at:
        try:
            posted_at = datetime.fromisoformat(posted_at)
        except Exception:
            posted_at = None
    apply_url = data.get("apply_url") or info.get("apply_url")
    features = data.get("features") or data.get("llm_features") or info.get("features")
    if not features:
        features = v7 or v5 or {}
    
    # Fix 5: Ensure inferred_titles is at top level of features
    if isinstance(features, dict) and "inferred_titles" not in features:
        if "inferred_titles" in v7:
            features["inferred_titles"] = v7["inferred_titles"]
        elif "inferred_titles" in v5:
            features["inferred_titles"] = v5["inferred_titles"]

    # Normalize country from location (e.g., "... , US , ...")
    if isinstance(features, dict) and location:
        loc_parts = [p.strip() for p in location.split(",")]
        for part in loc_parts[::-1]:
            upper = part.upper()
            if upper in {"US", "USA"}:
                features["country"] = "US"
                break

    return {
        "id": data.get("id") or data.get("job_id"),
        "title": title,
        "description": description,
        "company_name": company_name or "Unknown",
        "location": location,
        "remote": remote,
        "org_type": org_type,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "posted_at": posted_at,
        "apply_url": apply_url,
        "features": features,
    }


async def load_jobs(path: Path, batch_size: int, limit: int | None = None) -> None:
    jobs = []
    skipped_missing_embeddings = 0
    processed = 0
    async with SessionLocal() as session:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit and processed >= limit:
                    break
                data = _parse_line(line)
                if not data:
                    continue
                processed += 1
                job_info = _extract_job_info(data)
                if not job_info["id"]:
                    continue
                embedding_explicit = _get_embedding(data, "embedding_explicit")
                embedding_inferred = _get_embedding(data, "embedding_inferred")
                embedding_company = _get_embedding(data, "embedding_company")
                if not embedding_explicit or not embedding_inferred or not embedding_company:
                    skipped_missing_embeddings += 1
                    continue
                job_info["embedding_explicit"] = embedding_explicit
                job_info["embedding_inferred"] = embedding_inferred
                job_info["embedding_company"] = embedding_company
                jobs.append(job_info)
                if len(jobs) >= batch_size:
                    stmt = pg_insert(Job).values(jobs).on_conflict_do_nothing(index_elements=["id"])
                    await session.execute(stmt)
                    await session.commit()
                    jobs = []
            if jobs:
                stmt = pg_insert(Job).values(jobs).on_conflict_do_nothing(index_elements=["id"])
                await session.execute(stmt)
                await session.commit()
    print(
        f"Processed {processed} lines, skipped {skipped_missing_embeddings} missing embeddings."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to jobs.jsonl")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(load_jobs(Path(args.path), args.batch_size, args.limit))


if __name__ == "__main__":
    main()
