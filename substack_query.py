#!/usr/bin/env python3
import os # Add os import for API key and DB URL
import sys # Add sys import for exiting on critical errors
import argparse # Add argparse for command-line arguments

# Fix Windows console encoding issues with Unicode characters - must be done early
if sys.platform == "win32":
    # Set UTF-8 encoding for the environment
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # Reconfigure stdout/stderr with UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import requests
import json
from datetime import datetime, timedelta, timezone, date
import re
from markdownify import markdownify # Add markdownify import
import psycopg2 # Add psycopg2 import
from psycopg2 import extras # Import extras for batch insertion
from dotenv import load_dotenv # Add dotenv import
from pgvector.psycopg2 import register_vector # Add pgvector import
import feedparser # Add feedparser import
from urllib.parse import urlparse # Add urlparse import
from bs4 import BeautifulSoup # Add BeautifulSoup import
import time # Add time import
import logging # Add logging import
from collections import deque # Add deque import

# --- Environment Variables ---
load_dotenv(override=True) # Load .env file and override system env vars

# Shared Azure/OpenAI helpers: one structured analysis call per post, relevance gate, embeddings.
from llm_common import (
    LLMContentFiltered, LLMError, analyze_content, build_embedding_text,
    generate_embeddings, is_ai_safety_content, check_llm_or_exit,
)

LOCAL_PROXY_RE = re.compile(r"^https?://(?:127\.0\.0\.1|localhost)(?::\d+)?/?$", re.IGNORECASE)

def disable_unreachable_local_proxies() -> None:
    """
    Disable loopback proxy env vars that can break feed/API requests.
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

# --- Top Level Buffer ---
SKIPPED_BUFFER = deque()   # (post_id, title_norm, url)

# --- Helper for title normalization ---
def normalise_title(t: str) -> str:
    """Normalize title: lowercase, replace multiple spaces with single, strip leading/trailing."""
    if not isinstance(t, str): return "" # Handle non-string input
    return re.sub(r'\s+', ' ', t).strip().lower()

def clean_title_for_storage(title: str | None) -> str | None:
    """Create a stable display-safe title variant for the cleaned_title column."""
    if not title:
        return None
    return re.sub(r'\s+', ' ', title).strip()

def buffer_skip(post_id, title, url):
    """Add a post to the skip buffer, avoiding duplicates."""
    title_norm = normalise_title(title or "")
    # Only append if this title_norm isn't already in the buffer
    if title_norm not in {t for _, t, _ in SKIPPED_BUFFER}:
        SKIPPED_BUFFER.append(
            (str(post_id) if post_id else "N/A",
             title_norm,
             url)
        )

def flush_skip_buffer(db_url: str) -> list:
    """
    Insert buffered gate/cutoff/paywall skips in their OWN transaction, so they
    persist even when this run accepts no posts. Returns the rows written.
    """
    if not SKIPPED_BUFFER:
        return []
    rows = list(SKIPPED_BUFFER)
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            extras.execute_values(
                cur,
                "INSERT INTO skipped_posts (post_id, title_norm, source_url) VALUES %s ON CONFLICT DO NOTHING",
                rows,
            )
        conn.commit()
    finally:
        conn.close()
    SKIPPED_BUFFER.clear()
    logging.info("Recorded %d cutoff/guard-rail/paywall skips.", len(rows))
    return rows

# Setup logging early, before any potential logging calls
# Use WARNING level by default to reduce verbosity, but keep errors visible
# Allow DEBUG mode via environment variable for troubleshooting
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"
# LOG_LEVEL=INFO shows per-post progress; DEBUG=true is more verbose still.
log_level = logging.DEBUG if DEBUG_MODE else getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

if DEBUG_MODE:
    print("[DEBUG] DEBUG mode enabled - verbose logging active")

# Suppress overly verbose logs from underlying libraries if desired
logging.getLogger("urllib3").setLevel(logging.ERROR) # Added for requests/urllib3 noise
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR) # Suppress OpenAI logs if needed
disable_unreachable_local_proxies()

# --- Essential Environment Variables ---
DATABASE_URL = os.environ.get("AI_SAFETY_FEED_DB_URL")

# --- Initial Checks ---
if not DATABASE_URL:
    logging.critical("CRITICAL ERROR: AI_SAFETY_FEED_DB_URL environment variable not set. Cannot connect to database.")
    sys.exit(1) # Exit if DB URL is missing
check_llm_or_exit(embeddings=True)  # env check + one preflight call; aborts on a bad deployment/key

# --- Constants ---
BATCH_SIZE = 1 # Size for batch database inserts

# Define the cutoff date (inclusive) - posts before this date are ignored
CUTOFF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)

# List of Substack feeds
SUBSTACK_FEEDS = [
    "https://agifriday.substack.com",
    "https://aifrontiersmedia.substack.com",
    "https://www.safeai.news",
    "https://newsletter.safe.ai",
    "https://www.astralcodexten.com", # Note: Changed from .substack.com to .com
    "https://thezvi.substack.com",
    "https://oliverpatel.substack.com",
    "https://epochai.substack.com",
    "https://artificialintelligenceact.substack.com",
    "https://www.hyperdimensional.co",
    "https://joecarlsmith.substack.com",
    "https://milesbrundage.substack.com",
    "https://newsletter.mlsafety.org",
    "https://aligned.substack.com",
    "https://helentoner.substack.com",
]

# Mapping for friendly source names based on Substack host/slug
SUBSTACK_SOURCE_NAMES = {
    "agifriday.substack.com"           : "AGI Friday",
    "aifrontiersmedia.substack.com"    : "AI Frontiers",
    "safeai.news"                      : "AI Safety & Governance Newsletter",
    "newsletter.safe.ai"               : "AI Safety Newsletter", 
    "astralcodexten.com"               : "Astral Codex Ten", 
    "thezvi.substack.com"              : "Don't Worry About the Vase",
    "oliverpatel.substack.com"         : "Enterprise AI Governance",
    "epochai.substack.com"             : "Epoch AI",
    "artificialintelligenceact.substack.com": "The EU AI Act Newsletter",
    "hyperdimensional.co"              : "Hyperdimensional",
    "joecarlsmith.substack.com"        : "Joe Carlsmith's Substack",
    "milesbrundage.substack.com"       : "Miles's Substack",
    "newsletter.mlsafety.org"          : "ML Safety Newsletter",
    "aligned.substack.com"             : "Musings on the Alignment Problem",
    "helentoner.substack.com"          : "Rising Tide",
}

# API clients (Azure chat + OpenAI embeddings) are created lazily by llm_common.
print("[OK] LLM configuration loaded (see llm_common.py)")

# --- Database Columns and Pre-computed INSERT statement ---
# ================================================================
#                      Database Configuration
# ================================================================
# Define the columns in the 'content' table that we will insert into
# IMPORTANT: Ensure this order matches the data tuple created later.
# 'title_norm' is excluded as it's a generated column in the DB.
DB_COLS = (
    "source_url", "title", "source_type", "authors", "published_date",
    "topics", "score", "image_url", "sentence_summary", "paragraph_summary",
    "key_implication", "full_content", "full_content_markdown",
    "comment_count", "cluster_tag",
    "embedding_short", "embedding_full",
    "cleaned_title",
)
# Calculate the number of columns dynamically
NUM_DB_COLS = len(DB_COLS)
# Pre-compute the INSERT SQL statement for efficiency
# The f-string logic automatically adjusts the number of '%s' placeholders
INSERT_SQL = f"""
INSERT INTO content ({', '.join(DB_COLS)})
VALUES ({', '.join(['%s'] * NUM_DB_COLS)})
ON CONFLICT (title_norm) DO NOTHING;
"""

# --- Helper SQL & function for skipping posts ---
SKIP_INSERT_SQL = """
INSERT INTO skipped_posts (post_id, title_norm, source_url)
VALUES (%s, %s, %s)
ON CONFLICT DO NOTHING;
"""

def record_skip(cur, post_id: str, title_norm: str, source_url: str | None):
    """
    Insert one row into skipped_posts and keep the caller side-effect-free
    (no commit here – caller decides when to commit or roll back).
    """
    cur.execute(SKIP_INSERT_SQL,
                (str(post_id) if post_id else "N/A",
                 str(title_norm),
                 source_url))

# --- Relevance gate (shared implementation in llm_common) ---
GATE_SNIPPET_BYTES = 8000

# Recurring non-article formats (open threads, link roundups, forecasting digests,
# comment highlights). They mention AI often enough to fool the LLM gate on a short
# snippet, so reject them by title before spending a model call.
NON_ARTICLE_TITLE_RE = re.compile(
    r"^\s*(open (hidden )?(open )?thread|mantic monday|links for \w+|model city monday|"
    r"highlights from the comments|meetups everywhere|classifieds thread|berkeley meetup)",
    re.IGNORECASE,
)

def is_ai_safety_post(title: str, html_body: str, max_bytes: int = GATE_SNIPPET_BYTES) -> bool | None:
    """
    Fast yes/no guard-rail on the title + start of the post.
    Returns True/False from the model, or None when the model could not be
    reached. Callers must treat None as "unknown": do not ingest AND do not
    record a skip, so the post is retried on the next run.
    """
    if title and NON_ARTICLE_TITLE_RE.search(title):
        logging.info(f"Guard-rail: rejecting recurring non-article format by title: '{title[:60]}'")
        return False
    snippet = markdownify(html_body or "", heading_style="ATX", bullets='-')
    snippet = snippet.encode('utf-8')[:max_bytes].decode('utf-8', 'ignore')
    return is_ai_safety_content(title, snippet, kind="Substack post")

# --- Substack Fetching Helpers ---

# Helper to safely convert value to int, defaulting to 0
def safe_int_or_zero(value):
    """Safely converts value to int, defaulting to 0 if None, invalid, or TypeError."""
    if value is None: return 0
    try: return int(value)
    except (ValueError, TypeError): return 0

# Helper to parse ISO date strings reliably
def iso_to_dt(iso_string: str) -> datetime | None:
    """Converts ISO 8601 string to timezone-aware datetime object (UTC)."""
    if not iso_string:
        return None
    try:
        # Handle 'Z' suffix for UTC
        dt_obj = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        # Ensure timezone is UTC if naive
        if dt_obj.tzinfo is None:
            return dt_obj.replace(tzinfo=timezone.utc)
        # Convert to UTC if it has another timezone
        return dt_obj.astimezone(timezone.utc)
    except (ValueError, TypeError):
        logging.warning(f"Could not parse ISO date string: '{iso_string}'")
        return None

# Helper to extract Substack slug from URL or pass through if already a slug
def slug_from_thing(source: str) -> str | None:
    """
    Accepts a Substack root URL or bare 'slug' and returns the canonical host
    used in API endpoints, e.g. 'agifriday.substack.com' or 'safeai.news'.
    """
    if not source or not isinstance(source, str):
        return None

    source = source.strip()
    parsed = urlparse(source if "://" in source else f"https://{source}")

    host = parsed.netloc or parsed.path.split("/")[0]  # handles bare slugs
    if not host:
        return None

    # Keep the host exactly as configured. Some publications only serve the API
    # on the www. host (astralcodexten.com and safeai.news answer 404/406 on the
    # apex), and str.lstrip("www.") strips characters, not a prefix.
    return host.lower()

def display_slug(slug: str) -> str:
    """Slug without a leading 'www.' (used for author/source display)."""
    return slug[4:] if slug.startswith("www.") else slug

def source_name_for(slug: str) -> str:
    """Friendly source name; SUBSTACK_SOURCE_NAMES is keyed without 'www.'."""
    return SUBSTACK_SOURCE_NAMES.get(display_slug(slug)) or SUBSTACK_SOURCE_NAMES.get(slug) or display_slug(slug)

# Recursive helper to extract text from body_json nodes
def extract_text(node):
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return extract_text(node.get("text") or node.get("children") or "")
    if isinstance(node, list):
        # Use newline for lists to better separate paragraphs/blocks
        return "\n\n".join(extract_text(c) for c in node)
    return ""

# Helper to convert Substack JSON API response to our standard post dict
def json_to_post(full: dict, slug: str) -> dict | None:
    """Converts the JSON object from Substack's post API to our standard dict format."""
    post_id = full.get("id")
    if not post_id:
        logging.warning(f"Substack post JSON missing 'id' field for slug '{slug}'. Skipping.")
        return None # ID is essential

    # Try 'published_at' first, then 'post_date', then 'updated_at'
    posted_at_dt = iso_to_dt(
        full.get("published_at")
     or full.get("post_date")
     or full.get("updated_at")
    )
    if not posted_at_dt:
        logging.warning(f"Substack post JSON missing or invalid 'published_at'/'post_date'/'updated_at' for ID {post_id}, slug '{slug}'. Skipping.")
        return None # Date is essential

    # Skip pay-walled content
    # Note: This check is done here based on the full post details
    if full.get("audience") == "only_paid" and full.get("should_show_paywall"):
        # logging.info(f"Skipping paywalled post ID {post_id} for slug '{slug}'.") # Optional info log
        return None

    # Calculate score with multiple fallbacks
    score = safe_int_or_zero(
        full.get("reaction_value") or
        full.get("public_reactions_count") or
        safe_int_or_zero(sum((full.get("reactions") or {}).values())) # Safely sum reactions
    )

    # Extract HTML body with fallbacks (including body_json and truncated_body_text)
    html_body = (
        full.get("body_html")
        or full.get("body_markdown")
        # Updated: Use extract_text helper for body_json
        or ("\n\n".join(extract_text(p) for p in full.get("body_json", [])))
        or (f"<p>{full['truncated_body_text']}</p>" if full.get('truncated_body_text') else "")
    )
    # Ensure it's never empty if description exists
    if not html_body and full.get("description"):
        html_body = f"<p>{full['description']}</p>" # Wrap description

    # Extract original tags from Substack post data (try both 'post_tags' and 'postTags')
    original_tags = []
    raw = full.get("post_tags") or full.get("postTags") or []
    for t in raw:
        # postTags items are dicts; post_tags items are strings
        if isinstance(t, dict):
            original_tags.append({"name": t.get("name")})
        elif isinstance(t, str):
            original_tags.append({"name": t})

    # --- Extract image URL (moved into json_to_post) ---
    image_url = None # Default to None
    # Try to find first image in HTML body
    if html_body:
        try:
            temp_soup = BeautifulSoup(html_body, 'html.parser')
            img_tag = temp_soup.find('img')
            if img_tag:
                # Prefer data-src if available, fallback to src
                image_url = img_tag.get('data-src') or img_tag.get('src')
        except Exception as e:
            logging.warning(f"Could not parse image from HTML body for post ID {post_id}: {e}")
    # Fallback: Use cover_image if no image found in HTML
    if not image_url:
        image_url = full.get("cover_image")
    # --- End image extraction ---

    return {
        "_id":        str(post_id), # Ensure string ID
        "title":      full.get("title", "Untitled"),
        "pageUrl":    full.get("canonical_url"),
        "postedAt":   posted_at_dt.isoformat(),
        "htmlBody":   html_body,
        "image_url":  image_url, # Add extracted image_url
        "tags":       original_tags, # Use extracted original tags
        "user":       {"displayName": display_slug(slug)}, # Use slug as a placeholder user/source identifier
        "coauthors":  [], # JSON API doesn't provide easily
        "baseScore":  score,
        "commentCount": full.get("comments_count", 0),
        # source_type will be added later using the mapping
    }


# ---------- Substack Iterators ----------

def iter_substack_archive_stubs(slug: str, cutoff: datetime, batch: int = 35):
    """
    Yields lightweight post stubs from the Substack Archive API for a given slug,
    stopping when posts are older than the cutoff date.
    This is the first phase - just get the basic info without full post content.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
    }
    offset = 0
    while True:
        list_url = f"https://{slug}/api/v1/archive?sort=new&search=&offset={offset}&limit={batch}"
        # Remove debug logging here
        try:
            page_response = requests.get(list_url, timeout=45, headers=headers)
            page_response.raise_for_status()
            json_response = page_response.json()
            if isinstance(json_response, list):
                stubs = json_response
            elif isinstance(json_response, dict):
                stubs = json_response.get("posts", [])
            else:
                logging.warning(f"Unexpected JSON response type ({type(json_response).__name__}) for '{slug}' at offset {offset}. Assuming empty.")
                stubs = []
            time.sleep(1)  # Shorter delay for list fetching
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logging.error(f"Failed fetching/decoding archive page for '{slug}' at offset {offset}: {e}")
            raise

        if not stubs:
            # Remove debug logging here
            break

        # Remove debug logging here
        reached_cutoff = False
        for stub in stubs:
            posted_at_dt = iso_to_dt(stub.get("post_date"))
            if not posted_at_dt:
                logging.warning(f"Skipping stub with missing/invalid 'post_date' in archive for '{slug}': ID {stub.get('id')}")
                continue
            if posted_at_dt < cutoff:
                # Remove debug logging here
                reached_cutoff = True
                break

            yield stub

        if reached_cutoff:
            break

        offset += len(stubs)

def fetch_single_post(slug: str, stub: dict, headers: dict):
    """
    Fetches a single full post from Substack API given a stub.
    Returns the converted post_dict, or one of the strings
    "fetch_failed" | "unusable" (paywalled / missing fields) | "rejected" (gate said no)
    | "gate_error" (gate unavailable; retry next run).
    """
    pid = stub.get("id")
    title = stub.get('title', 'Untitled')
    
    # Construct URL
    post_slug = stub.get("slug")
    if post_slug:
        full_post_url = f"https://{slug}/api/v1/posts/{post_slug}"
    else:
        full_post_url = f"https://{slug}/api/v1/posts/by-id/{pid}"
        
    try:
        full_response = requests.get(full_post_url, timeout=45, headers=headers)
        full_response.raise_for_status()
        full_post_data = full_response.json()
        time.sleep(2)  # Politeness delay
        
    except Exception as e:
        logging.warning(f"Failed fetching full post ID {pid} for '{slug}': {e}")
        return "fetch_failed"
        
    if not isinstance(full_post_data, dict):
        logging.warning(f"Substack API for post ID {pid} ('{slug}') returned unexpected type ({type(full_post_data).__name__}) instead of dict.")
        return "fetch_failed"
        
    # Convert to standard post dict
    post_dict = json_to_post(full_post_data, slug)
    if not post_dict:
        return "unusable"
        
    # AI safety check
    verdict = is_ai_safety_post(post_dict["title"], post_dict["htmlBody"])
    if verdict is None:
        # Model unavailable: don't ingest, don't blacklist; retry next run.
        logging.warning(f"Guard-rail unavailable for '{title[:50]}...' ({slug}); will retry next run.")
        return "gate_error"
    if not verdict:
        buffer_skip(pid, post_dict["title"], post_dict.get("pageUrl"))
        return "rejected"
        
    post_dict["source_type"] = source_name_for(slug)
    return post_dict

def iter_substack_rss(slug: str, cutoff: datetime, limit: int = 25, existing_titles: set = None, already_skipped: set = None):
    """
    Yields post objects parsed from a Substack RSS feed for a given slug,
    up to a limit and respecting the cutoff date.
    Handles feedparser errors gracefully.
    
    Args:
        slug: The Substack domain/slug
        cutoff: Earliest date to include posts from
        limit: Maximum number of posts to process
        existing_titles: Set of normalized titles already in content table (for early filtering)
        already_skipped: Set of normalized titles already in skipped_posts table (for early filtering)
    """
    # Initialize sets if not provided (backward compatibility)
    if existing_titles is None:
        existing_titles = set()
    if already_skipped is None:
        already_skipped = set()
    
    known_titles = existing_titles | already_skipped
    
    # slug may already be 'domain.com' or 'xyz.substack.com'
    feed_url = f"https://{slug}/feed"
    print(f"Fetching RSS: {slug}") # Keep this visible
    headers = { # Add headers dict
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
    }
    try:
        # Feedparser handles network errors internally to some extent
        # Pass request_headers to feedparser.parse
        d = feedparser.parse(feed_url, request_headers=headers)
        if d.bozo: # Check if feedparser encountered issues
             logging.warning(f"Error parsing RSS feed for '{slug}': {d.bozo_exception}")
             # Optionally return here if strict error handling is needed
             # return
    except Exception as e:
         # Catch unexpected errors during parsing setup
         logging.error(f"Unexpected error initializing feedparser for '{slug}': {e}")
         return # Stop iteration for this feed

    if not d or not d.entries:
        print(f"[WARN] RSS feed for '{slug}' is empty or could not be fetched.")
        return # Nothing to iterate

    count = 0
    for entry in d.entries:
        if count >= limit:
            break # Stop if limit reached

        # Basic sanity checks
        link = entry.get("link")
        title = entry.get("title")
        published_parsed = entry.get("published_parsed")
        if not link or not title or not published_parsed:
            logging.warning(f"Skipping RSS entry for '{slug}' due to missing link, title, or date: {link or 'No Link'}")
            continue

        # Parse date and check cutoff
        try:
            # feedparser returns struct_time, convert to datetime
            dt = datetime(*published_parsed[:6], tzinfo=timezone.utc)
        except (ValueError, TypeError):
            logging.warning(f"Skipping RSS entry for '{slug}' due to invalid date tuple: {published_parsed}")
            continue

        if dt < cutoff:
            # Remove debug logging here
            buffer_skip(entry.get("id", link), title, link)
            continue # Skip posts older than cutoff

        # --- EARLY DATABASE CHECK (NEW) ---
        title_norm = normalise_title(title)
        if title_norm in known_titles:
            # Remove debug logging here
            continue # Skip if already known, avoiding expensive AI safety check
        # --- END EARLY DATABASE CHECK ---

        # Extract content (prefer full content over summary)
        html_body = ""
        if entry.get("content") and isinstance(entry.get("content"), list) and len(entry.get("content")) > 0:
            html_body = entry.get("content")[0].get('value', '')
        elif entry.get("summary_detail"):
            html_body = entry.summary_detail.get('value', '')
        elif entry.get("summary"): # Fallback to basic summary
             html_body = entry.summary

        # Extract tags from RSS feed
        tag_objs = [{"name": t.term} for t in entry.tags if t and t.term] if "tags" in entry else []

        # Use link or id as fallback _id
        post_id = entry.get("id", link)

        post_dict = {
            "_id":        post_id, # Use ID or link
            "title":      title,
            "pageUrl":    link,
            "postedAt":   dt.isoformat(),
            "htmlBody":   html_body,
            "tags":       tag_objs, # Use tags parsed from RSS
            "user":       {"displayName": display_slug(slug)}, # Use slug as placeholder user/source
            "coauthors":  [{"displayName": entry.get("author")}] if entry.get("author") else [],
            "baseScore":  None,     # RSS has no score
            "commentCount": None,   # RSS has no comments
            "source_type": source_name_for(slug) # Use friendly name from mapping
        }

        # --- Guard-rail: Check if post is AI safety related (MOVED AFTER DB CHECK) ---
        verdict = is_ai_safety_post(post_dict["title"], post_dict["htmlBody"])
        if verdict is None:
            print(f"   [RETRY] Gate unavailable, will retry next run: '{title[:50]}...'")
            continue                      # not recorded as skipped
        if not verdict:
            print(f"   [SKIP] Not AI safety (gate): '{title[:50]}...'")
            buffer_skip(post_id, title, link)
            continue                      # discard & do NOT yield

        yield post_dict
        count += 1

# --- Substack Orchestration Function ---
def fetch_substack_optimized(source_input: str, cutoff: datetime, existing_titles: set, 
                           already_skipped: set, rss_limit: int = 25, 
                           archive_batch: int = 100, always_add_rss: bool = False):
    """
    Optimized version that checks database before fetching full posts.
    
    Args:
        source_input: Substack slug or full feed/publication URL.
        cutoff: The earliest date (inclusive) for posts.
        existing_titles: Set of normalized titles already in content table
        already_skipped: Set of normalized titles already in skipped_posts table
        rss_limit: Max posts to fetch via RSS fallback.
        archive_batch: Batch size for archive pagination.
        always_add_rss: If True, always fetches RSS (after archive) to catch newest posts.
    """
    slug = slug_from_thing(source_input)
    if not slug:
        logging.error(f"Cannot fetch Substack: Invalid source input '{source_input}'")
        return []
    
    # Combined set for faster lookups
    known_titles = existing_titles | already_skipped
    
    posts_to_fetch = []  # List of stubs that need full fetching
    posts_by_id = {}
    archive_failed = False
    
    print(f"\n[FETCH] Fetching from: {slug}")
    
    # STEP 1: Get all post stubs first (lightweight)
    try:
        for stub in iter_substack_archive_stubs(slug, cutoff=cutoff, batch=archive_batch):
            # Check title against database
            title = stub.get('title', '')
            if not title:
                continue
                
            title_norm = normalise_title(title)
            
            # Skip if already in database or previously skipped
            if title_norm in known_titles:
                # Remove debug logging here
                continue
                
            # This post needs to be fetched
            posts_to_fetch.append(stub)
            
        print(f"   [OK] Found {len(posts_to_fetch)} new posts from archive")

    except Exception as e:
        print(f"   [ERROR] Archive API failed: {e}")
        logging.error(f"Archive API stub fetch FAILED for '{slug}': {e}. Will attempt RSS fallback.")
        archive_failed = True
    
    # STEP 2: Only fetch full details for posts not in database
    if not archive_failed and posts_to_fetch:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            )
        }
        
        successful_fetches = 0
        outcomes = {"rejected": 0, "gate_error": 0, "unusable": 0, "fetch_failed": 0}
        for i, stub in enumerate(posts_to_fetch):
            title = stub.get('title', 'Untitled')
            # Only log every 5th post or failures to reduce verbosity
            if (i + 1) % 5 == 0 or i == len(posts_to_fetch) - 1:
                print(f"   [INFO] Fetching posts... [{i+1}/{len(posts_to_fetch)}]")
            
            result = fetch_single_post(slug, stub, headers)
            if isinstance(result, dict) and result.get('_id'):
                posts_by_id[result['_id']] = result
                successful_fetches += 1
            elif result == "rejected":
                outcomes["rejected"] += 1
                print(f"   [SKIP] Not AI safety (gate): '{title[:50]}...'")
            elif result == "gate_error":
                outcomes["gate_error"] += 1
                print(f"   [RETRY] Gate unavailable, will retry next run: '{title[:50]}...'")
            elif result == "unusable":
                outcomes["unusable"] += 1
                # Paywalled / missing fields is permanent: record it so we stop re-fetching.
                buffer_skip(stub.get("id"), title, stub.get("canonical_url"))
                print(f"   [SKIP] Paywalled or missing fields: '{title[:50]}...'")
            else:
                outcomes["fetch_failed"] += 1
                print(f"   [FAIL] Failed to fetch: '{title[:50]}...'")

        print(f"   [OK] Archive fetch complete: {successful_fetches} kept, "
              f"{outcomes['rejected']} rejected by gate, {outcomes['unusable']} paywalled/unusable, "
              f"{outcomes['fetch_failed']} fetch failures, {outcomes['gate_error']} gate errors "
              f"(of {len(posts_to_fetch)})")
        
    elif archive_failed:
        print(f"   → Will use RSS fallback")
    else:
        print(f"   → No new posts found in archive")
    
    # STEP 3: Handle RSS if needed (with similar pre-filtering)
    need_rss = archive_failed or always_add_rss
    if need_rss:
        reason = "fallback" if archive_failed else "always_add_rss"
        print(f"   [RSS] Fetching RSS ({reason})...")
        try:
            rss_count = 0
            # Use existing RSS iterator but with pre-filtering
            for post in iter_substack_rss(slug, cutoff=cutoff, limit=rss_limit, existing_titles=existing_titles, already_skipped=already_skipped):
                if post and post.get('_id'):
                    # Check if this title is already known before adding
                    title_norm = normalise_title(post.get('title', ''))
                    if title_norm not in known_titles:
                        posts_by_id.setdefault(post['_id'], post)
                        rss_count += 1
                    # Remove debug logging here
            print(f"   [OK] RSS complete: {rss_count} new posts")
        except Exception as e:
            print(f"   [ERROR] RSS failed: {e}")
            logging.error(f"RSS fetch FAILED for '{slug}': {e}")

    # STEP 4: Return sorted results
    final_posts = list(posts_by_id.values())
    final_posts.sort(key=lambda p: p.get('postedAt', ''), reverse=True)

    print(f"   [SUMMARY] Total for {slug}: {len(final_posts)} posts")
    return final_posts

def fetch_substack(source_input: str, cutoff: datetime, rss_limit: int = 25, archive_batch: int = 100, always_add_rss: bool = False):
    """
    DEPRECATED: Use fetch_substack_optimized() instead.
    
    Backward compatibility wrapper that fetches posts for a Substack publication.
    This version is less efficient as it doesn't check the database before fetching full posts.
    """
    logging.warning("fetch_substack() is deprecated. Consider using fetch_substack_optimized() for better performance.")
    # Use optimized version with empty sets (no pre-filtering)
    return fetch_substack_optimized(
        source_input=source_input,
        cutoff=cutoff,
        existing_titles=set(),
        already_skipped=set(),
        rss_limit=rss_limit,
        archive_batch=archive_batch,
        always_add_rss=always_add_rss
    )

# --- Filtering Logic (Removed EA/LW/AF specific filters) ---
# No specific filtering logic needed here anymore, as Substack fetching
# already handles cutoff date and the AI safety guard-rail.
# Score-based filtering could be added here if desired for Substack posts.

# --- De-duplication Helper ---
def choose_highest_score(posts):
    """Keeps only the highest-scoring post for each normalized title."""
    by_title = {}
    # Filter out posts without titles first, as they cannot be deduplicated
    valid_posts = [p for p in posts if p and p.get('title')]
    original_valid_count = len(valid_posts)
    print(f"\n[DEDUP] Deduplicating {original_valid_count} posts by title...")

    removed_count = 0
    for p in valid_posts:
        # Normalize title: lowercase, replace multiple spaces with single, strip leading/trailing
        key = normalise_title(p['title'])
        # Treat None score (possible from RSS) as 0 for comparison purposes
        current_score = p.get('baseScore') or 0

        existing_entry = by_title.get(key)

        # If no entry exists for this title, add the current post
        if existing_entry is None:
            by_title[key] = p
        else:
            # If an entry exists, compare scores (treating None as 0)
            existing_score = existing_entry.get('baseScore') or 0
            # Replace if the current post has a strictly higher score
            if current_score > existing_score:
                # Remove debug logging here
                by_title[key] = p
                removed_count += 1 # Count the one that was replaced
            else:
                # Remove debug logging here
                removed_count += 1 # Count the one being discarded

    unique_posts = list(by_title.values())
    # Calculate removed duplicates based on the difference from the initial valid count
    # duplicates_removed = original_valid_count - len(unique_posts) # Old calculation was slightly off

    print(f"   [OK] Kept {len(unique_posts)} unique posts (removed {removed_count} duplicates)")
    return unique_posts

# --- Helper for title normalization ---
def normalise_title(t: str) -> str:
    """Normalize title: lowercase, replace multiple spaces with single, strip leading/trailing."""
    if not isinstance(t, str): return "" # Handle non-string input
    return re.sub(r'\s+', ' ', t).strip().lower()

# --- Backfill Function for Missing Analysis ---

def backfill_missing_analysis(batch_size=10):
    """
    Backfill missing sentence_summary, paragraph_summary, and key_implication
    for existing Substack posts that have content but are missing analysis.
    
    Args:
        batch_size: Number of rows to process per run
    """
    print(f"\n{'='*60}")
    print("BACKFILL MODE: Re-analyzing existing Substack posts")
    print(f"{'='*60}\n")
    
    conn = None
    processed_count = 0
    updated_count = 0
    skipped_count = 0
    
    try:
        print("[DB] Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        print("[OK] Database connected")
        register_vector(conn)
        
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            # Find Substack posts missing analysis
            print(f"[INFO] Searching for Substack posts missing analysis...")
            cur.execute("""
                SELECT id, title, source_type, sentence_summary, paragraph_summary, 
                       key_implication, full_content_markdown, full_content, topics, cluster_tag
                FROM content
                WHERE source_type = ANY(%s)
                AND (sentence_summary IS NULL OR paragraph_summary IS NULL OR key_implication IS NULL 
                     OR topics IS NULL OR cluster_tag IS NULL)
                AND (full_content_markdown IS NOT NULL OR full_content IS NOT NULL)
                ORDER BY published_date DESC NULLS LAST
                LIMIT %s
            """, (list(SUBSTACK_SOURCE_NAMES.values()), batch_size))
            
            rows = cur.fetchall()
            
            if not rows:
                print("✓ No Substack posts need backfilling - all caught up!")
                return
            
            print(f"[OK] Found {len(rows)} Substack posts to process\n")
            
            # Process each row
            for i, row in enumerate(rows, 1):
                row_id = row['id']
                title = row['title'] or 'Untitled'
                
                try:
                    print(f"[{i}/{len(rows)}] Processing: '{title[:60]}...'")
                    print(f"   ID: {row_id}")
                    print(f"   Source: {row['source_type']}")
                    
                    # Determine what content to use
                    content = row['full_content_markdown'] or row['full_content']
                    if not content or content.isspace():
                        print(f"   [SKIP] No usable content")
                        skipped_count += 1
                        continue
                    
                    # Track what needs updating
                    updates = {}
                    needs_embeddings = False
                    
                    # One structured call regenerates every analysis field; keep only the missing ones.
                    try:
                        analysis = analyze_content(title, content, [])
                    except (LLMError, ValueError) as e:
                        print(f"   ✗ Analysis failed: {e}")
                        skipped_count += 1
                        continue
                    if row['sentence_summary'] is None:
                        updates['sentence_summary'] = analysis['sentence_summary']
                        needs_embeddings = True
                        print(f"   ✓ Sentence summary generated")
                    if row['paragraph_summary'] is None:
                        updates['paragraph_summary'] = analysis['paragraph_summary']
                        needs_embeddings = True
                        print(f"   ✓ Paragraph summary generated")
                    if row['key_implication'] is None:
                        updates['key_implication'] = analysis['key_implication']
                        needs_embeddings = True
                        print(f"   ✓ Key implication generated")
                    if row['cluster_tag'] is None:
                        updates['cluster_tag'] = analysis['cluster_tag']
                        needs_embeddings = True
                        print(f"   ✓ Cluster tag generated: {analysis['cluster_tag']}")
                    if row['topics'] is None:
                        updates['topics'] = analysis['tags']
                        needs_embeddings = True
                        print(f"   ✓ Topics generated: {analysis['tags']}")

                    # Regenerate embeddings if we updated any summaries
                    if needs_embeddings:
                        print(f"   → Regenerating embeddings...")
                        
                        # Get current values (use newly generated or existing)
                        sentence_sum = updates.get('sentence_summary') or row['sentence_summary'] or ""
                        paragraph_sum = updates.get('paragraph_summary') or row['paragraph_summary'] or ""
                        key_impl = updates.get('key_implication') or row['key_implication'] or ""
                        topics = updates.get('topics') or row['topics'] or []
                        
                        # Prepare embedding texts
                        text_for_short = title
                        text_for_full = build_embedding_text(sentence_sum, paragraph_sum, key_impl, topics if isinstance(topics, list) else [])
                        
                        # Generate embeddings
                        emb_short, emb_full = generate_embeddings(text_for_short, text_for_full)
                        
                        if emb_short and emb_full:
                            updates['embedding_short'] = emb_short
                            updates['embedding_full'] = emb_full
                            print(f"   ✓ Embeddings regenerated")
                        else:
                            print(f"   ⚠ Embedding generation failed; not saving summaries so the row is retried")
                            skipped_count += 1
                            continue
                    
                    # Update database if we have changes
                    if updates:
                        set_clauses = []
                        values = []
                        
                        for key, value in updates.items():
                            set_clauses.append(f"{key}=%s")
                            values.append(value)
                        
                        values.append(row_id)  # For WHERE clause
                        
                        update_sql = f"UPDATE content SET {', '.join(set_clauses)} WHERE id=%s"
                        cur.execute(update_sql, values)
                        conn.commit()
                        
                        print(f"   ✓ Database updated ({len(updates)} fields)")
                        updated_count += 1
                        processed_count += 1
                    else:
                        print(f"   [SKIP] No successful updates generated")
                        skipped_count += 1
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                    print()  # Blank line between posts
                    
                except Exception as e:
                    print(f"   ✗ Error processing row {row_id}: {e}")
                    logging.error(f"Failed to process row {row_id}: {e}", exc_info=True)
                    if conn:
                        conn.rollback()
                    skipped_count += 1
                    print()
            
            print(f"\n{'='*60}")
            print("BACKFILL SUMMARY")
            print(f"{'='*60}")
            print(f"Posts found:       {len(rows)}")
            print(f"Successfully updated: {updated_count}")
            print(f"Skipped:           {skipped_count}")
            print(f"{'='*60}\n")
    
    except psycopg2.Error as db_err:
        print(f"\n✗ Database error: {db_err}")
        logging.error(f"Database error: {db_err}", exc_info=True)
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("[OK] Database connection closed")

# --- Main Execution & Database Handling ---

def main(limit: int | None = None):
    # limit: process at most N new posts (canary runs)
    # Setup logging (already done at the top)
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Suppress overly verbose logs from underlying libraries (already done at the top)
    # logging.getLogger("google.generativeai").setLevel(logging.WARNING)
    # logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Check for Database URL first
    if not DATABASE_URL:
        logging.error("Error: DATABASE_URL environment variable not set. Cannot connect to database.")
        sys.exit(1) # Exit with error code

    # --- Connect to DB and fetch existing titles FIRST ---
    conn = None
    existing_titles = set()
    already_skipped = set()
    
    try:
        print("\n[DB] Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        print("[OK] Database connected")
        register_vector(conn) # Register pgvector type with the connection
        print("[OK] pgvector registered")

        with conn.cursor() as cur:
            # Fetch existing titles BEFORE processing any feeds
            print("[INFO] Fetching existing data...")
            cur.execute("SELECT title_norm FROM content")
            existing_titles = {row[0] for row in cur.fetchall()}
            print(f"   → {len(existing_titles):,} titles in content table")

            # Fetch already skipped titles
            cur.execute("SELECT title_norm FROM skipped_posts")
            already_skipped = {row[0] for row in cur.fetchall()}
            logging.info(f"→ {len(already_skipped):,} titles already in skipped_posts table")

    except psycopg2.OperationalError as e:
        logging.critical(f"FATAL: Database connection failed: {e}")
        sys.exit(1) # Exit with error code
    except psycopg2.DatabaseError as e:
        logging.error(f"Database error occurred while fetching existing titles: {e}")
        sys.exit(1) # Exit with error code
    except Exception as e:
        logging.error(f"An unexpected error occurred while connecting to database: {e}", exc_info=True)
        sys.exit(1) # Exit with error code
    finally:
        if conn:
            conn.close()
            logging.info("Initial database connection closed.")

    # --- Fetch Substack Data with Database Info ---
    print("\n[START] Starting Substack feed fetch...")
    substack_posts_all = []
    # Option to always fetch RSS as a safety net for very new posts
    # Set this via env var or keep False
    ALWAYS_FETCH_RSS_NET = os.getenv("SUBSTACK_ALWAYS_FETCH_RSS", "False").lower() == "true"

    early_skips = []
    for feed_index, feed_url_or_slug in enumerate(SUBSTACK_FEEDS, 1):
        print(f"\n[FEED {feed_index}/{len(SUBSTACK_FEEDS)}] {feed_url_or_slug}", flush=True)
        # Use the optimized fetch_substack function with database info
        posts_for_feed = fetch_substack_optimized(
            source_input=feed_url_or_slug,
            cutoff=CUTOFF_DATE,
            existing_titles=existing_titles,
            already_skipped=already_skipped,
            rss_limit=40, # Max RSS posts per feed if needed
            archive_batch=20, # Archive API page size
            always_add_rss=ALWAYS_FETCH_RSS_NET
        )
        substack_posts_all.extend(posts_for_feed)
        # Persist this feed's gate/cutoff/paywall skips right away (own transaction), so a
        # killed run does not lose them and progress is visible in the DB.
        flushed = flush_skip_buffer(DATABASE_URL)
        early_skips.extend(flushed)
        already_skipped.update(t for _, t, _ in flushed)
        print(f"   [DB] {len(flushed)} skips recorded for this feed; {len(substack_posts_all)} candidate posts so far", flush=True)

    early_skip_count = len(early_skips)

    initial_fetched_count = len(substack_posts_all)
    print(f"\n[COMPLETE] Fetch complete: {initial_fetched_count} posts total")

    # --- Deduplicate in memory before processing ---
    unique_posts = choose_highest_score(substack_posts_all)
    total_unique_count = len(unique_posts)

    if not unique_posts:
        print(f"[INFO] No new posts to process ({early_skip_count} skips recorded).")
        return

    # --- Process Posts and Insert into Database ---
    processed_count = 0
    affected_rows_count = 0 # Tracks total rows inserted
    failed_analysis_count = 0 # Track posts where analysis failed/skipped
    total_failures = 0 # Track failures during batch insert
    skipped_in_db_count = 0 # Track skipped due to being in DB
    skipped_missing_data_count = 0 # Track skipped due to missing URL/Title
    skipped_analysis_error_count = 0 # Track skipped due to HTML/Markdown error
    new_posts_started = 0 # For --limit
    total_skipped_by_record_count = early_skip_count # Track posts recorded in skipped_posts table (incl. skips flushed above)

    conn = None # Initialize conn outside the try block

    try:
        # Establish a new connection for processing
        logging.info("\n--- Database Operations ---")
        logging.info("Connecting to the database for processing...")
        conn = psycopg2.connect(DATABASE_URL)
        logging.info("Database connection successful.")
        register_vector(conn) # Register pgvector type with the connection
        logging.info("pgvector type registered with psycopg2 connection.")

        with conn.cursor() as cur:
            logging.info(f"\n--- Processing {total_unique_count} Unique Posts ---")
            posts_to_insert = 0 # Count posts added to batch_data
            for i, post in enumerate(unique_posts): # Iterate over unique posts
                processed_count += 1
                analysis_successful = True # Flag to track if all analyses succeed for this post

                # Safely get values, providing defaults
                url = post.get('pageUrl', 'N/A')
                title = post.get('title', 'N/A') # Get the raw title
                post_id = post.get('_id', 'Unknown ID')

                logging.info(f"\n[{i+1}/{total_unique_count}] Processing Post: '{title[:60]}...' ({url})")

                # --- Skip if URL is invalid or title missing (early checks) ---
                title_norm = normalise_title(title) # Normalize title once for checks
                if url == 'N/A' or title == 'N/A':
                     logging.warning(f"  -> Skipping post (ID: {post_id}): Missing URL or Title.")
                     record_skip(cur, post_id, title_norm, url); conn.commit() # Record skip
                     already_skipped.add(title_norm) # Add to in-memory cache
                     total_skipped_by_record_count += 1 # Increment counter
                     skipped_missing_data_count += 1
                     continue # Skip to the next post

                # --- Normalize title and check against existing DB titles (EARLY OUT) ---
                # Note: This check should be much faster now since most titles were already filtered out
                if title_norm in existing_titles:
                    logging.info(f"  -> Skipping post (already in DB): '{title[:60]}...'")
                    skipped_in_db_count += 1
                    continue # Already in DB, skip all further processing for this post
                # --- End early-out check ---

                # --- Check against already_skipped titles (EARLY OUT) ---
                if title_norm in already_skipped:
                    logging.info("  → Skipping (known bad): %s", title[:60])
                    # No counter increment here as it's already been recorded and counted in this run or previous
                    continue
                # --- End already_skipped check ---

                if limit is not None:
                    if new_posts_started >= limit:
                        logging.warning(f"  --limit {limit} reached; stopping before this post.")
                        print(f"[INFO] --limit {limit} reached; stopping.")
                        break
                    new_posts_started += 1

                # --- Check for invalid publication date before extensive processing ---
                published_date_str = post.get('postedAt', '')
                published_date_dt = None
                if published_date_str:
                    try:
                        published_date_dt = datetime.fromisoformat(published_date_str)
                        # Check against CUTOFF_DATE (though iterators should handle this, good for safety)
                        if published_date_dt < CUTOFF_DATE:
                            logging.warning(f"  -> Skipping post (ID: {post_id}): Publication date {published_date_dt.date()} is before cutoff {CUTOFF_DATE.date()}.")
                            record_skip(cur, post_id, title_norm, url); conn.commit()
                            already_skipped.add(title_norm)
                            total_skipped_by_record_count += 1
                            continue
                    except ValueError:
                        logging.warning(f"  -> Skipping post (ID: {post_id}): Invalid publication date string '{published_date_str}'.")
                        record_skip(cur, post_id, title_norm, url); conn.commit()
                        already_skipped.add(title_norm)
                        total_skipped_by_record_count += 1
                        continue
                else: # No 'postedAt' field
                    logging.warning(f"  -> Skipping post (ID: {post_id}): Missing publication date ('postedAt').")
                    record_skip(cur, post_id, title_norm, url); conn.commit()
                    already_skipped.add(title_norm)
                    total_skipped_by_record_count += 1
                    continue
                # --- End publication date check ---

                # --- Extract original tags for use in cluster analysis ---
                tags_list = post.get('tags', []) # These are the original tags from Substack
                tag_names = [tag.get('name', 'N/A') for tag in tags_list if tag and isinstance(tag.get('name'), str)]

                source_type = post.get('source_type', 'Unknown') # Get added source_type

                score = safe_int_or_zero(post.get('baseScore'))
                comment_count = post.get('commentCount') # Keep as number (or None) for DB
                full_content = post.get('htmlBody', '') or "" # Ensure it's a string

                # --- Prepend title to HTML content ---
                if title != 'N/A' and full_content: # Only add if title exists and content exists
                    full_content = f"<h1>{title}</h1>\n\n{full_content}" # Use \n\n for markdown friendliness later
                # --- End prepend title ---

                # --- Clean HTML with BeautifulSoup (Moved after duplicate check) ---
                cleaned_html = ""
                if full_content:
                    try:
                        soup = BeautifulSoup(full_content, 'html.parser')
                        for element in soup(["script", "style", "iframe", "form", "button", "input"]): # Remove unwanted tags
                            element.decompose()
                        # Optional: Add more specific cleaning here if needed
                        cleaned_html = str(soup)
                    except Exception as e:
                        logging.error(f"ERROR: BeautifulSoup cleaning failed for post '{title}' ({url}). Error: {e}. Content and analysis will be skipped.")
                        cleaned_html = "" # Keep it empty on error
                        analysis_successful = False # Skip analysis if cleaning fails
                        record_skip(cur, post_id, title_norm, url); conn.commit() # Record skip
                        already_skipped.add(title_norm) # Add to in-memory cache
                        total_skipped_by_record_count += 1 # Increment counter
                        skipped_analysis_error_count += 1 # Count this specific skip reason
                else:
                    logging.debug(f"  -> No HTML content found for post '{title[:60]}...', skipping cleaning.") # Info if content was empty
                    # This case implies full_content was empty. If this is a reason to skip, record it.
                    # Assuming empty HTML means we should skip and record.
                    logging.warning(f"  -> Skipping post (ID: {post_id}): HTML content is empty.")
                    record_skip(cur, post_id, title_norm, url); conn.commit()
                    already_skipped.add(title_norm)
                    total_skipped_by_record_count += 1
                    skipped_analysis_error_count += 1 # Or a more specific counter for empty content
                    analysis_successful = False # Mark as false if no content to analyze
                    # Don't increment skipped_count here, might still insert basic info

                # --- Convert Cleaned HTML to Markdown ---
                full_content_markdown = ""
                # Only proceed if HTML cleaning was successful (or if original content was empty but cleaning wasn't skipped)
                if analysis_successful and cleaned_html:
                    try:
                        # Use the cleaned HTML for markdown conversion
                        logging.debug(f"  -> Converting cleaned HTML to Markdown...")
                        full_content_markdown = markdownify(cleaned_html, heading_style="ATX", bullets="-")
                        logging.debug(f"  -> Markdown conversion successful.")
                    except Exception as e:
                        logging.error(f"ERROR: Could not convert cleaned HTML to Markdown for post '{title}' ({url}). Error: {e}. Content and analysis will be skipped.")
                        full_content_markdown = "" # Keep it empty on error
                        analysis_successful = False # Skip analysis if markdown fails
                        # Only count skip if cleaning didn't already fail
                        if cleaned_html is not None:
                            record_skip(cur, post_id, title_norm, url); conn.commit() # Record skip
                            already_skipped.add(title_norm) # Add to in-memory cache
                            total_skipped_by_record_count += 1 # Increment counter
                            skipped_analysis_error_count += 1
                elif not full_content and analysis_successful: # Handle case where original content was empty
                    logging.debug(f"  -> Original content was empty, Markdown is also empty.")
                    # This implies cleaned_html was also empty or not processed.
                    # If markdown is empty and it's a skip condition, it should have been caught by HTML check.
                    # However, if cleaning succeeded but produced empty markdown from non-empty HTML (unlikely with markdownify),
                    # or if we want to explicitly skip if full_content_markdown is empty after processing.
                    if not full_content_markdown and analysis_successful: # analysis_successful means HTML was processed
                        logging.warning(f"  -> Skipping post (ID: {post_id}): Markdown content is empty after conversion.")
                        record_skip(cur, post_id, title_norm, url); conn.commit()
                        already_skipped.add(title_norm)
                        total_skipped_by_record_count += 1
                        skipped_analysis_error_count += 1
                        analysis_successful = False

                else:
                     # Handle cases where markdown was empty or conversion failed earlier
                     if url != 'N/A' and skipped_analysis_error_count == 0 and skipped_missing_data_count == i + 1 - skipped_in_db_count: # Only log if not already counted as skipped
                         logging.warning(f"  -> Skipping analysis (empty/failed content): '{title[:60]}...'")
                     # analysis_successful should already be False here
                     # If analysis_successful is false at this point due to content issues,
                     # and it hasn't been recorded yet, record it.
                     if not analysis_successful and title_norm not in already_skipped:
                         logging.warning(f"  -> Recording skip for (ID: {post_id}) due to prior content processing failure.")
                         record_skip(cur, post_id, title_norm, url); conn.commit()
                         already_skipped.add(title_norm)
                         total_skipped_by_record_count += 1
                         # Note: skipped_analysis_error_count might have already been incremented.
                         # Avoid double counting for the print summary if an earlier specific error was caught.

                if not analysis_successful and skipped_analysis_error_count == 0 and skipped_missing_data_count == i + 1 - skipped_in_db_count: # Only count if not already skipped
                     failed_analysis_count += 1 # Increment if any analysis failed/skipped for this post

                # --- If analysis_successful is false at this point, we should not proceed to DB insertion logic.
                # The existing logic for batch_data.append should be conditional on analysis_successful OR
                # if we intend to insert posts with only basic info even if analysis failed.
                # The user request is to record skip for "empty or unparsable HTML / Markdown",
                # which implies these posts are not inserted.
                # Let's add a continue if analysis_successful is false after all content checks.

                if not analysis_successful:
                    logging.warning(f"  -> Final check: Skipping DB preparation for (ID: {post_id}) as analysis_successful is false.")
                    # Ensure it was recorded if not already. This is a safeguard.
                    if title_norm not in already_skipped:
                        record_skip(cur, post_id, title_norm, url); conn.commit()
                        already_skipped.add(title_norm)
                        total_skipped_by_record_count += 1
                    continue # Skip to the next post if analysis failed

                # --- AI Analysis Block ---
                logging.info(f"  -> Starting AI analysis for '{title[:60]}...'")
                sentence_summary = None
                paragraph_summary = None
                key_implication = None
                db_cluster = None
                db_tags = [] # Default for ARRAY type
                embedding_short_vector = None
                embedding_full_vector = None

                # One structured call returns all analyses (see llm_common.analyze_content).
                # Any failure means the post is NOT inserted: a transient error leaves the
                # title out of both tables so the next run retries it; a content-filter
                # rejection is recorded in skipped_posts so we stop retrying it.
                if not full_content_markdown or full_content_markdown.isspace():
                    logging.warning(f"  -> Skipping '{title[:60]}...': markdown content is empty. Recording skip.")
                    record_skip(cur, post_id, title_norm, url); conn.commit()
                    already_skipped.add(title_norm)
                    total_skipped_by_record_count += 1
                    failed_analysis_count += 1
                    continue

                try:
                    analysis = analyze_content(title, full_content_markdown, tag_names)
                except LLMContentFiltered as e:
                    logging.warning(f"  -> Content filter blocked '{title[:60]}...': {e}. Recording skip.")
                    record_skip(cur, post_id, title_norm, url); conn.commit()
                    already_skipped.add(title_norm)
                    total_skipped_by_record_count += 1
                    failed_analysis_count += 1
                    continue
                except (LLMError, ValueError) as e:
                    logging.error(f"  -> Analysis failed for '{title[:60]}...': {e}. Not inserted; will retry next run.")
                    failed_analysis_count += 1
                    continue

                sentence_summary = analysis["sentence_summary"]
                paragraph_summary = analysis["paragraph_summary"]
                key_implication = analysis["key_implication"]
                db_cluster = analysis["cluster_tag"]
                db_tags = analysis["tags"]

                embedding_short_vector, embedding_full_vector = generate_embeddings(
                    title if title and title != 'N/A' else "",
                    build_embedding_text(sentence_summary, paragraph_summary, key_implication, db_tags),
                )
                if embedding_short_vector is None or embedding_full_vector is None:
                    logging.error(f"  -> Embedding generation failed for '{title[:60]}...'. Not inserted; will retry next run.")
                    failed_analysis_count += 1
                    continue

                logging.info(f"  -> AI analysis finished for '{title[:60]}...'. Cluster: {db_cluster}, Tags: {db_tags}")
                # --- End AI Analysis Block ---

                # --- Extract other data ---
                # Image URL is now extracted directly in json_to_post
                image_url = post.get('image_url') # Get image_url added by json_to_post

                # Extract authors (Substack often just has the publication name/slug)
                post_authors_set = set()
                # Use 'user' displayName (slug) as primary author placeholder
                if post.get('user') and post['user'].get('displayName'):
                    post_authors_set.add(post['user']['displayName'])
                # Add coauthors if present (less common in Substack data structure)
                if post.get('coauthors'):
                    for author in post['coauthors']:
                        if author and author.get('displayName'):
                            post_authors_set.add(author['displayName'])
                authors_list = sorted(list(post_authors_set)) # Convert set to sorted list for ARRAY type
                # Ensure authors_list is never empty for the database
                if not authors_list:
                    authors_list = ['Unknown'] # Default if somehow empty

                # Extract publication date
                published_date_str = post.get('postedAt', '')
                published_date = None # Use None for DB if date is invalid
                if published_date_str:
                    try:
                        # Already timezone-aware from iso_to_dt
                        published_date = datetime.fromisoformat(published_date_str)
                    except ValueError:
                        logging.warning(f"Could not parse final datetime '{published_date_str}' for DB insertion for '{title[:50]}...'")
                        pass # Keep as None

                # AI-generated tags (db_tags) are used for the 'topics' column
                # AI-generated cluster (db_cluster) is used for the 'cluster_tag' column

                # --- Prepare data tuple for batch insertion ---
                logging.debug(f"  -> Preparing data tuple for DB insertion for post ID {post_id}.")
                # Ensure order matches DB_COLS
                cleaned_title = None  # left NULL on insert; rewrite_titles.py --mode titles fills it (frontend/backend fall back to title)
                data_tuple = (
                    url,                            # source_url
                    title,                          # title
                    source_type,                    # source_type
                    authors_list,                   # authors (list for ARRAY type)
                    published_date,                 # published_date (datetime or None)
                    db_tags,                        # topics (AI generated tags or None)
                    score,                          # score (int or None)
                    image_url,                      # image_url (str or None)
                    sentence_summary,               # sentence_summary (str or None)
                    paragraph_summary,              # paragraph_summary (str or None)
                    key_implication,                # key_implication (str or None)
                    cleaned_html,                   # full_content (cleaned HTML or None)
                    full_content_markdown,          # full_content_markdown (str or None)
                    comment_count,                  # comment_count (int)
                    db_cluster,                     # cluster_tag (AI generated cluster or None)
                    embedding_short_vector,         # embedding_short (list[float] or None)
                    embedding_full_vector,          # embedding_full (list[float] or None)
                    cleaned_title,                  # cleaned_title (str or None)
                )
                posts_to_insert += 1

                # --- Insert this post (one row per transaction) ---
                try:
                    cur.execute(INSERT_SQL, data_tuple)
                    conn.commit()
                    affected_rows_count += 1
                    existing_titles.add(title_norm)
                    logging.info(f"  -> DB insert OK: '{title[:60]}...'")
                except psycopg2.DatabaseError as e:
                    logging.error(f"  -> DB insert FAILED for '{title[:60]}...': {e}")
                    conn.rollback()
                    total_failures += 1

    except psycopg2.OperationalError as e:
        logging.critical(f"FATAL: Database connection failed: {e}")
        sys.exit(1) # Exit with error code
    except psycopg2.DatabaseError as e:
        logging.error(f"Database error occurred: {e}")
        if conn:
            conn.rollback() # Rollback any potential changes
        sys.exit(1) # Exit with error code
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main processing loop: {e}", exc_info=True) # Log traceback
        if conn:
            conn.rollback() # Rollback on general errors too
        sys.exit(1) # Exit with error code
    finally:
        if conn:
            conn.close() # Ensure connection is closed
            logging.info("Database connection closed.")

    print(f"\n--- Processing Summary ---")
    print(f"Total Substack posts fetched (initial): {initial_fetched_count}")
    print(f"Unique posts after deduplication:       {total_unique_count}")
    print(f"Posts processed:                        {processed_count}")
    print(f"Posts skipped (missing URL/Title):      {skipped_missing_data_count}")
    print(f"Posts skipped (already in DB):          {skipped_in_db_count}")
    print(f"Posts skipped (content/analysis error): {skipped_analysis_error_count}")
    print(f"Posts recorded in skipped_posts table:  {total_skipped_by_record_count}") # Add new counter to summary
    print(f"Posts with analysis failures (Gem/Emb): {failed_analysis_count}") # Posts attempted analysis but failed
    print(f"Posts prepared for DB insertion:        {posts_to_insert}")
    print(f"DB rows affected (estimate):            {affected_rows_count}") # Estimate based on batch success/retries
    print(f"DB insert failures (individual rows):   {total_failures}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Substack AI Safety Feed Scraper - Fetch and analyze Substack posts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal mode - fetch new posts:
  python substack_query.py
  
  # Backfill mode - re-analyze existing posts missing analysis:
  python substack_query.py --backfill
  
  # Backfill with custom batch size:
  python substack_query.py --backfill --batch-size 20
        """
    )
    
    parser.add_argument(
        '--backfill',
        action='store_true',
        help='Backfill missing analysis for existing Substack posts instead of fetching new posts'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process at most N new posts (canary runs)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of posts to process in backfill mode (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Run in appropriate mode
    if args.backfill:
        backfill_missing_analysis(batch_size=args.batch_size)
    else:
        main(limit=args.limit)
