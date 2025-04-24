#!/usr/bin/env python3
"""
One‑off utility to compute percentile break‑points and raw tag frequencies for the
AI‑safety content aggregator.  It intentionally skips *all* Gemini analysis,
DB writes, and heavy data‑cleaning — it just fetches the latest raw posts,
crunches the numbers, and exits.

What it does
============
1. Fetch the most‑recent posts from:
   • EA Forum (AI‑safety tag)
   • LessWrong (AI tag)
   • Alignment Forum (front page "top" view)
2. Print the 90‑th‑percentile score & comment count for each forum.
3. Build a *case‑insensitive* tag‑frequency table (before any downstream
   filtering) and write the tags that appear ≥2 times to
   `tag_frequency_report.csv`.

Because this script is only run on‑demand you can be generous with
timeouts — it does not need to be super fast.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable, List, Dict, Any

import requests
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
#  Config & constants  ────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # .env must be present in the working directory

EA_API_URL = "https://forum.effectivealtruism.org/graphql"
EA_AI_SAFETY_TAG_ID = "oNiQsBHA3i837sySD"

LW_API_URL = "https://www.lesswrong.com/graphql"
LW_AI_SAFETY_TAG_ID = "yBXKqk8wEg6eM8w5y"

AF_API_URL = "https://www.alignmentforum.org/graphql"

DEFAULT_LIMIT = 3000
CUTOFF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)

APRIL_FOOLS_TAGS = {"April Fool's", "April Fools' Day"}
AI_TAGS_LW = {"AI"}
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
#  Utility helpers  ────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def percentile(values: List[int | float], p: float) -> float:
    """Return the *p*‑th percentile of *values* (0‑100)."""
    if not values:
        return math.nan
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def report_percentiles(posts: List[dict[str, Any]], name: str, p: int = 90) -> None:
    if not posts:
        print(f"{name}: no data")
        return
    scores = [post.get("baseScore") or 0 for post in posts]
    comments = [post.get("commentCount") or 0 for post in posts]
    print(
        f"{name} – {p}‑th percentile: "
        f"score ≈ {percentile(scores, p):.0f}, "
        f"comments ≈ {percentile(comments, p):.0f}"
    )


def accumulate_tags(posts: Iterable[dict[str, Any]]) -> Counter[str]:
    """Return *Counter({tag_name: occurrences})* (case‑insensitive)."""
    c: Counter[str] = Counter()
    for post in posts:
        seen: set[str] = set()
        for tag in post.get("tags", []):
            if not tag:
                continue
            tag_name = (tag.get("name") or "").strip()
            if not tag_name:
                continue
            tag_key = tag_name.lower()
            if tag_key in seen:
                continue
            seen.add(tag_key)
            c[tag_name] += 1  # keep original case for display
    return c

# ──────────────────────────────────────────────────────────────────────────────
#  Data fetchers  ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def _run_graphql_query(api_url: str, query: str) -> dict[str, Any] | None:
    try:
        resp = requests.post(api_url, json={"query": query}, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            print(f"GraphQL errors from {api_url}: {json.dumps(data['errors'], indent=2)}")
            return None
        return data.get("data", {})
    except Exception as e:
        print(f"Request to {api_url} failed: {e}")
        return None


def get_forum_posts(api_url: str, tag_id: str | None = None, limit: int = DEFAULT_LIMIT) -> List[dict[str, Any]]:
    post_fields = """
          _id
          title
          pageUrl
          commentCount
          baseScore
          postedAt
          tags { _id name }
    """
    view_clause = f'view: "tagById"\n tagId: "{tag_id}"' if tag_id else 'view: "top"'
    query = f"""
    {{
      posts(input: {{ terms: {{ {view_clause}\n limit: {limit} }} }}) {{
        results {{ {post_fields} }}
      }}
    }}"""

    data = _run_graphql_query(api_url, query)
    if not data:
        return []
    return data.get("posts", {}).get("results", [])

# ──────────────────────────────────────────────────────────────────────────────
#  Main logic  ─────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1) Pull posts ------------------------------------------------------------------
    print("Fetching forums …")
    ea_posts = get_forum_posts(EA_API_URL, EA_AI_SAFETY_TAG_ID)
    lw_posts = get_forum_posts(LW_API_URL, LW_AI_SAFETY_TAG_ID)
    af_posts = get_forum_posts(AF_API_URL)

    # 2) Percentile report -----------------------------------------------------------
    print("\n=== Percentile thresholds (90th) ===")
    report_percentiles(ea_posts, "EA Forum")
    report_percentiles(lw_posts, "LessWrong")
    report_percentiles(af_posts, "Alignment Forum")
    print("====================================\n")

    # 3) Tag frequency --------------------------------------------------------------
    raw_tag_counts: Counter[str] = Counter()
    for bucket in (ea_posts, lw_posts, af_posts):
        raw_tag_counts.update(accumulate_tags(bucket))

    popular_tags = {t: n for t, n in raw_tag_counts.items() if n > 1}

    print("=== TOP 25 TAGS USED ≥2 TIMES (unfiltered feed) ===")
    if popular_tags:
        sorted_tags = sorted(popular_tags.items(), key=lambda t: (-t[1], t[0].lower()))
        for tag, n in sorted_tags[:25]:  # Limit to top 25
            print(f"{tag}: {n}")
    else:
        print("No tag appeared more than once.")
    print("=================================================\n")

    # 4) CSV output -----------------------------------------------------------------
    out_name = "tag_frequency_report.csv"
    try:
        with open(out_name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Tag", "Count"])
            for tag, n in sorted(popular_tags.items(), key=lambda t: (-t[1], t[0].lower())):
                writer.writerow([tag, n])
        print(f"Wrote {len(popular_tags)} rows to {out_name}")
    except IOError as e:
        print(f"ERROR: could not write {out_name}: {e}")


if __name__ == "__main__":
    main()
