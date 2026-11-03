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
  • EXIT_* codes the stage scripts return to run_pipeline.py.
  • RetryState: per-row failure counts (logs/retry_state.json) so rows that keep
    failing are skipped instead of starving the backlog.

The parked scripts (arxiv, twitter) and archive/ keep their own copies on purpose.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Iterable

import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")

# Exit codes the stage scripts return to run_pipeline.py (instead of matching stdout).
EXIT_DONE = 0        # finished, or nothing left to do
EXIT_FAILED = 1      # fatal error, or every item attempted this run failed
EXIT_MORE_WORK = 3   # a full batch was processed and more rows may remain: run again


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


# ────────────────────────── Poison-row tracking ──────────────────────────
RETRY_STATE_PATH = os.path.join(LOG_DIR, "retry_state.json")
MAX_ROW_FAILURES = 3


class RetryState:
    """
    Per-stage failure counts for content rows, kept in logs/retry_state.json
    ({stage: {row_id: failures}}). The backfill/novelty loops always pick the newest
    unfinished rows, so a row that fails every time would be re-selected forever and
    starve older ones. Rows with MAX_ROW_FAILURES or more failures are skipped (and
    logged). A success clears the row. Delete the file (or the stage's entry) to give
    skipped rows another try, e.g. after an API outage.
    """

    def __init__(self, stage: str, path: str = RETRY_STATE_PATH):
        self.stage = stage
        self.path = path
        self._all: dict[str, dict[str, int]] = {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._all = {k: v for k, v in data.items() if isinstance(v, dict)}
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            logging.warning("Could not read %s (%s); starting with empty retry state.", path, e)
        self.counts: dict[str, int] = self._all.setdefault(stage, {})

    def skip_ids(self) -> list[int]:
        """Row ids that have failed MAX_ROW_FAILURES or more times in this stage (logged)."""
        ids = sorted(int(k) for k, n in self.counts.items() if n >= MAX_ROW_FAILURES)
        if ids:
            logging.warning("Skipping %d row(s) that failed %d+ times in %s: %s (see %s)",
                            len(ids), MAX_ROW_FAILURES, self.stage, ids, os.path.basename(self.path))
        return ids

    def record(self, failed: Iterable = (), succeeded: Iterable = ()) -> None:
        """Count one failure for each failed id, clear succeeded ids, and save."""
        for rid in failed:
            self.counts[str(rid)] = self.counts.get(str(rid), 0) + 1
        for rid in succeeded:
            self.counts.pop(str(rid), None)
        self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._all, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)
