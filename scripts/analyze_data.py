import json
import binascii
from pathlib import Path

def _parse_line(line: str) -> dict:
    line = line.strip()
    if not line: return {}
    if line.startswith("{"): return json.loads(line)
    try:
        decoded = binascii.unhexlify(line).decode("utf-8")
        return json.loads(decoded)
    except: return {}

def find_all_matches(d, target, path=""):
    results = []
    if isinstance(d, dict):
        for k, v in d.items():
            current_path = f"{path}.{k}" if path else k
            if target in str(k).lower() or target in str(v).lower():
                results.append((current_path, v))
            if isinstance(v, (dict, list)):
                results.extend(find_all_matches(v, target, current_path))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            current_path = f"{path}[{i}]"
            if target in str(v).lower():
                results.append((current_path, v))
            if isinstance(v, (dict, list)):
                results.extend(find_all_matches(v, target, current_path))
    return results

def analyze(path: str, limit: int = 2000):
    print(f"Analyzing first {limit} lines of {path}...")
    
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            data = _parse_line(line)
            if not data: continue
            
            remotes = find_all_matches(data, "remote")
            if remotes:
                # Filter out obvious noisy matches like location names if needed
                print(f"LINE {i} found 'remote' matches: {remotes[:2]}...")
            
            nonprofits = find_all_matches(data, "non-profit")
            if nonprofits:
                print(f"LINE {i} found 'non-profit' matches: {nonprofits[:2]}...")

            startups = find_all_matches(data, "startup")
            if startups:
                print(f"LINE {i} found 'startup' matches: {startups[:2]}...")

if __name__ == "__main__":
    analyze("/Users/la/Desktop/cafe/jobs.jsonl", limit=1000)
