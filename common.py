#!/usr/bin/env python3
"""
common.py
─────────
Non-LLM helpers shared by the pipeline scripts (ea_lw_query, podcast_query,
substack_query, the backfills, novelty_analysis, rewrite_titles). LLM clients
and calls live in llm_common.py.

What lives here:
  • get_db_connection(): one psycopg2 connection from AI_SAFETY_FEED_DB_URL.
  • normalise_title(): the title_norm used for dedup. It MUST keep matching the
    existing content.title_norm values (whitespace collapsed, lowercased).
  • record_skip(): one row into skipped_posts.
  • iso_to_dt(), disable_unreachable_local_proxies().

The parked scripts (arxiv, twitter) and archive/ keep their own copies on purpose.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)

HERE = os.path.dirname(os.path.abspath(__file__))


# ────────────────────────── Database ──────────────────────────
def get_db_connection(*, register_vectors: bool = False):
    """psycopg2 connection to AI_SAFETY_FEED_DB_URL (optionally with the pgvector adapter)."""
    db_url = (os.environ.get("AI_SAFETY_FEED_DB_URL") or "").strip()
    if not db_url:
        raise RuntimeError("AI_SAFETY_FEED_DB_URL is not set")
    conn = psycopg2.connect(db_url)
    if register_vectors:
        from pgvector.psycopg2 import register_vector
        register_vector(conn)
    return conn


SKIP_INSERT_SQL = """
INSERT INTO skipped_posts (post_id, title_norm, source_url)
VALUES (%s, %s, %s)
ON CONFLICT DO NOTHING;
"""


def record_skip(cur, post_id, title_norm: str, source_url: str | None) -> None:
    """Insert one row into skipped_posts (no commit; the caller decides)."""
    cur.execute(SKIP_INSERT_SQL, (
        str(post_id) if post_id else "N/A",
        str(title_norm),
        str(source_url) if source_url is not None else None,
    ))


# ────────────────────────── Text / dates ──────────────────────────
def normalise_title(title: str | None) -> str:
    """Lowercase, collapse runs of whitespace to one space, strip. Matches content.title_norm."""
    if not title or not isinstance(title, str):
        return ""
    return re.sub(r"\s+", " ", title).strip().lower()


def iso_to_dt(iso_string: str | None) -> datetime | None:
    """ISO 8601 string ('Z' allowed) -> timezone-aware UTC datetime; naive input is taken as UTC."""
    if not iso_string:
        return None
    try:
        dt_obj = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        if dt_obj.tzinfo is None:
            return dt_obj.replace(tzinfo=timezone.utc)
        return dt_obj.astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse ISO date string: '{iso_string}'. Error: {e}")
        return None


# ────────────────────────── Network ──────────────────────────
LOCAL_PROXY_RE = re.compile(r"^https?://(?:127\.0\.0\.1|localhost)(?::\d+)?/?$", re.IGNORECASE)


def disable_unreachable_local_proxies() -> None:
    """
    Disable loopback proxy env vars that can break outbound fetches in sandboxed runs.
    Keeps non-local proxy settings untouched.
    """
    proxy_keys = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    )
    removed = []
    for key in proxy_keys:
        value = os.environ.get(key)
        if value and LOCAL_PROXY_RE.match(value.strip()):
            os.environ.pop(key, None)
            removed.append(f"{key}={value}")
    if removed:
        logging.warning("Disabled local proxy env vars for network fetches: %s", ", ".join(removed))
