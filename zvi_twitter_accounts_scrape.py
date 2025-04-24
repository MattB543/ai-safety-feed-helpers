#!/usr/bin/env python3
"""
Scrape every post on thezvi.substack.com published on/after 2024-04-01 UTC,
extract all x.com / twitter.com links, tally the mentioned usernames,
and write results to links.csv.
"""

from __future__ import annotations

import csv, json, time, re, sys
from datetime import datetime, timezone
from typing import Any
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse         # Only needed if you later want hostnames

SLUG        = "thezvi"
CUTOFF      = datetime(2024, 4, 1, tzinfo=timezone.utc)
ARCH_BATCH  = 35          # posts per archive page
HEADERS     = {
    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
}

# Regex to capture the username (first path part) from Twitter/X URLs,
# ignoring common non-profile paths. Username must start with a letter or _.
TWITTER_RE = re.compile(
    r'https?://(?:www\.)?(?:x|twitter)\.com/'
    r'([a-zA-Z_]\w+)'  # Capture group 1: the username
    r'(?:/|/status/\d+|$)') # Must be followed by /, /status/, or end of string

# ---------------------------------------------------------------------------

def get_with_backoff(url: str, *, session: requests.Session, max_tries=6) -> requests.Response:
    """
    GET with exponential-backoff on 429 or 5xx.
    Respects the server's Retry-After header when present.
    """
    delay = 3.0  # Start delay at 3 seconds
    for attempt in range(1, max_tries + 1):
        resp = session.get(url, headers=HEADERS, timeout=45)
        if resp.status_code not in (429, 502, 503):
            resp.raise_for_status()
            return resp                # success
        # -- rate-limited or transient error --
        retry_after = resp.headers.get("retry-after")
        sleep_for   = float(retry_after) if retry_after and retry_after.isdigit() else delay
        print(f" → {resp.status_code} on {url} – sleeping {sleep_for:.1f}s (attempt {attempt}/{max_tries})",
              file=sys.stderr)
        time.sleep(sleep_for)
        delay *= 2                     # exponential
    resp.raise_for_status()            # still failing → bubble up

# ---------------------------------------------------------------------------

def parse_when(value: Any) -> datetime | None:
    """
    Accept *any* timestamp format Substack currently emits:
      • ISO-8601 string (with or without trailing Z)
      • Unix epoch milliseconds (int or str that .isdigit())
    Returns a UTC-aware datetime or None.
    """
    if value is None:
        return None

    # 1) milliseconds since the epoch
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        try:
            return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
        except (OSError, ValueError):
            return None

    # 2) ISO-8601 string
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None

    return None

def iter_archive(slug: str, cutoff: datetime, batch: int = 40):
    """
    Yield full-post JSON objects from Substack's archive API until < cutoff.
    Fetches via /api/v1/posts/{slug} (new), with id fallback.
    """
    offset, limit = 0, batch
    last_failed_url: str | None = None

    with requests.Session() as sess:           # ← one TCP session, reused
        while True:
            url = f"https://{slug}.substack.com/api/v1/archive?sort=new&search=&offset={offset}&limit={limit}"
            try:
                print(f"Fetching archive: {url}", file=sys.stderr)
                # 1) fetch archive page
                resp = get_with_backoff(url, session=sess)
                last_failed_url = None # Clear last failed URL on success
                time.sleep(0.5) # Add a small delay after successful archive fetch
            except requests.HTTPError as e:
                # Handle potential non-retryable errors after backoff attempts
                print(f"Failed to fetch {url} after multiple retries: {e}", file=sys.stderr)
                # Optionally, decide if this is fatal or if you want to try skipping
                # For now, let's re-raise to indicate a persistent problem
                raise

            stubs = resp.json() if isinstance(resp.json(), list) else resp.json().get("posts", [])
            if not stubs:
                print("No more stubs – done.", file=sys.stderr)
                break

            for stub in stubs:
                pid   = stub.get("id", "N/A")
                when  = parse_when(
                    stub.get("published_at")
                    or stub.get("published_at_local")
                    or stub.get("post_date")
                    or stub.get("post_date_local")
                    or stub.get("published_at_override")
                )
                if when is None:
                    print(f"Stub {pid} has no recognised timestamp – skipping.", file=sys.stderr)
                    continue
                if when < cutoff:
                    print(f"Post {pid} ({when}) < cutoff – stopping.", file=sys.stderr)
                    return

                post_slug = stub.get("slug")
                full_url  = (
                    f"https://{slug}.substack.com/api/v1/posts/{post_slug}"
                    if post_slug else
                    f"https://{slug}.substack.com/api/v1/post/{pid}"
                )

                try:
                    # 2) fetch full post
                    full = get_with_backoff(full_url, session=sess)
                    time.sleep(0.5) # Add a small delay after successful post fetch
                except requests.HTTPError as e:
                    # Final fallback: skip stubborn 404s and keep rolling
                    # Note: 429/5xx are handled by get_with_backoff now
                    if e.response is not None and e.response.status_code == 404:
                        print(f" → 404 for {full_url} – skipping.", file=sys.stderr)
                        continue
                    # Re-raise other HTTP errors that weren't handled by backoff
                    print(f"Failed to fetch full post {full_url}: {e}", file=sys.stderr)
                    raise e

                yield full.json()

            offset += len(stubs)


# ---------------------------------------------------------------------------

def extract_twitter_usernames(html: str) -> list[str]:
    """Extracts Twitter/X usernames from profile or status links in HTML."""
    if not html:
        return []
    usernames = set() # Use a set to avoid duplicates within a single post
    soup = BeautifulSoup(html, "html.parser")

    # 1. Find usernames in href attributes of <a> tags
    for a in soup.find_all("a", href=True):
        match = TWITTER_RE.match(a["href"])
        if match:
            usernames.add(match.group(1))

    # 2. Find usernames in plain text (less common, but possible)
    text_content = soup.get_text(" ")
    for match in TWITTER_RE.finditer(text_content):
        usernames.add(match.group(1))

    return list(usernames) # Return as list

# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Scanning {SLUG}.substack.com posts since {CUTOFF.date()} …", file=sys.stderr)
    counts: dict[str, int] = {}

    for post in iter_archive(SLUG, CUTOFF, ARCH_BATCH):
        # Use the new function to get usernames
        usernames = extract_twitter_usernames(
            post.get("body_html") or post.get("body_markdown") or ""
        )
        for name in usernames:
            counts[name] = counts.get(name, 0) + 1

    if not counts:
        print("No x.com / twitter.com usernames found.", file=sys.stderr)
        return

    print(f"Found {len(counts):,} unique usernames – writing links.csv", file=sys.stderr)
    with open("links.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Update CSV header
        writer.writerow(["username", "count"])
        # Sort by count desc, then username asc
        for name, cnt in sorted(counts.items(), key=lambda t: (-t[1], t[0])):
            writer.writerow([name, cnt])

    print("Done!", file=sys.stderr)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
