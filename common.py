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
  • RetryState: per-row attempt counts (logs/retry_state.json) so rows that keep
    failing are skipped instead of starving the backlog.
  • write_json_atomic(): unique temp file + os.replace.

The parked scripts (arxiv, twitter) and archive/ keep their own copies on purpose.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
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
        # register_vector runs a query, which opens a transaction; end it so callers
        # can still set conn.autocommit (psycopg2 refuses inside a transaction).
        conn.commit()
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
_LOCK_WAIT_SEC = 10
_LOCK_STALE_SEC = 60


def _validated_state(data, path: str) -> dict[str, dict[str, int]]:
    """Keep only {stage: {"<int id>": <int count >= 0>}} entries; log and drop anything else."""
    if not isinstance(data, dict):
        logging.warning("%s is not a JSON object; ignoring it.", path)
        return {}
    clean: dict[str, dict[str, int]] = {}
    dropped = 0
    for stage, rows in data.items():
        if not isinstance(stage, str) or not isinstance(rows, dict):
            dropped += 1
            continue
        good = {}
        for rid, n in rows.items():
            if isinstance(rid, str) and rid.isascii() and rid.isdigit() and isinstance(n, int) and not isinstance(n, bool) and n >= 0:
                good[rid] = n
            else:
                dropped += 1
        clean[stage] = good
    if dropped:
        logging.warning("Dropped %d malformed entr%s from %s.", dropped, "y" if dropped == 1 else "ies", path)
    return clean


def write_json_atomic(path: str, data) -> None:
    """Write JSON to a unique temp file in the same folder, then os.replace it into place."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True, ensure_ascii=False)
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows: a reader has the target open for a moment
                if attempt == 9:
                    raise
                time.sleep(0.1)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


class RetryState:
    """
    Per-stage attempt counts for content rows, kept in logs/retry_state.json
    ({stage: {row_id: failed_attempts}}). The backfill/novelty loops always pick the
    newest unfinished rows, so a row that fails every time would be re-selected forever
    and starve older ones.

    Usage: begin(ids) BEFORE the external work (it is saved at once, so a row whose call
    hangs and gets the process killed at the stage deadline still counts), then
    succeeded(ids) after the row's update is committed. Rows with MAX_ROW_FAILURES or
    more unfinished attempts are skipped (and logged). Every write reloads the file and
    merges under a lock file, so concurrent runs don't lose each other's counts.
    Delete the file (or the stage's entry) to give skipped rows another try, e.g. after
    an API outage.
    """

    def __init__(self, stage: str, path: str = RETRY_STATE_PATH):
        self.stage = stage
        self.path = path
        self.counts: dict[str, int] = {}
        self._update(lambda rows: None)

    def _load(self) -> dict[str, dict[str, int]]:
        try:
            with open(self.path, encoding="utf-8") as f:
                return _validated_state(json.load(f), self.path)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            logging.warning("Could not read %s (%s); starting with empty retry state.", self.path, e)
            return {}

    def _acquire_lock(self) -> int | None:
        lock = self.path + ".lock"
        os.makedirs(os.path.dirname(lock) or ".", exist_ok=True)
        deadline = time.monotonic() + _LOCK_WAIT_SEC
        while True:
            try:
                return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(lock) > _LOCK_STALE_SEC:
                        os.remove(lock)  # left behind by a killed process
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    logging.warning("Could not lock %s within %ds; updating it unlocked.", lock, _LOCK_WAIT_SEC)
                    return None
                time.sleep(0.05)

    def _update(self, change) -> None:
        """Reload under the lock, apply change(rows_of_this_stage), save if anything changed."""
        fd = self._acquire_lock()
        try:
            data = self._load()
            rows = data.setdefault(self.stage, {})
            before = dict(rows)
            change(rows)
            if rows != before:
                write_json_atomic(self.path, data)
            self.counts = dict(rows)
        finally:
            if fd is not None:
                os.close(fd)
                try:
                    os.remove(self.path + ".lock")
                except OSError:
                    pass

    def skip_ids(self) -> list[int]:
        """Row ids with MAX_ROW_FAILURES or more unfinished attempts in this stage (logged)."""
        self._update(lambda rows: None)
        ids = sorted(int(k) for k, n in self.counts.items() if n >= MAX_ROW_FAILURES)
        if ids:
            logging.warning("Skipping %d row(s) that failed %d+ times in %s: %s (see %s)",
                            len(ids), MAX_ROW_FAILURES, self.stage, ids, os.path.basename(self.path))
        return ids

    def begin(self, ids: Iterable) -> None:
        """Count an attempt for each id now; it stays counted unless succeeded() clears it."""
        ids = [str(i) for i in ids]

        def change(rows):
            for rid in ids:
                rows[rid] = rows.get(rid, 0) + 1
        if ids:
            self._update(change)

    def succeeded(self, ids: Iterable) -> None:
        """Clear ids whose work has been committed."""
        ids = [str(i) for i in ids]

        def change(rows):
            for rid in ids:
                rows.pop(rid, None)
        if ids:
            self._update(change)


# ────────────────────────── Feeds ──────────────────────────
def parse_feed_response(resp):
    """feedparser.parse() for a requests response, passing its headers and final URL
    (as content-location) so relative links and enclosures resolve."""
    import feedparser
    headers = {k.lower(): v for k, v in resp.headers.items()}
    headers["content-location"] = resp.url
    return feedparser.parse(resp.content, response_headers=headers)


def is_bot_checkpoint(resp) -> bool:
    """True if resp is a CDN bot-challenge page (e.g. Vercel's "Security Checkpoint", which
    the EA Forum has served to scripts since Nov 2026) rather than a real API answer.
    Retrying doesn't help: the site has to allow the client."""
    if resp.status_code not in (403, 429):
        return False
    if resp.headers.get("x-vercel-mitigated") == "challenge":
        return True
    return ("text/html" in resp.headers.get("Content-Type", "")
            and "Security Checkpoint" in resp.text[:5000])


def is_duplicate_episode_url(link: str | None, audio_url: str | None, stored_audio_by_url: dict) -> bool:
    """
    True if a podcast entry's link is already stored as some row's source_url AND one of the
    rows with that link has the same audio URL. stored_audio_by_url maps source_url -> set of
    audio URLs: a link alone can be a show homepage shared by many episodes, so without the
    audio match the caller falls back to the title_norm check.
    """
    if not link or not audio_url:
        return False
    return audio_url in stored_audio_by_url.get(link, ())
