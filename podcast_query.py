"""
Ingest podcast episodes from various RSS feeds relevant to AI Safety
into the `content` table of the AI-Safety-Feed database.

This script fetches episodes via RSS, transcribes audio with Gemini
(gemini_transcribe.py, Files API) if an audio URL is present, and falls back
to HTML show notes if transcription fails or is unavailable. It filters episodes based on date
and an AI safety guardrail, performs analysis (summarization, implication
identification, clustering/tagging via Azure OpenAI chat; embeddings via OpenAI)
on the available content (transcript preferred), and inserts the processed
data into a PostgreSQL database, handling duplicates based on normalized titles.

LLM analysis (one structured call per episode), the relevance gate and
embeddings live in llm_common.py and are shared with the other scrapers.
"""

# ================================================================
#                            Imports
# ================================================================
import os
import re
import json
import time
import logging
from datetime import datetime, timezone, timedelta
import sys
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any, Tuple, Set

import feedparser
from bs4 import BeautifulSoup
from markdownify import markdownify
import psycopg2
from psycopg2 import extras # For batch insertion
from psycopg2 import OperationalError as Psycopg2OpError # Alias for clarity
from pgvector.psycopg2 import register_vector # <-- ADDED for pgvector
from dotenv import load_dotenv

# --- AI/ML helpers (shared) ---
from llm_common import (
    LLMContentFiltered, LLMError, analyze_content, build_embedding_text,
    generate_embeddings, is_ai_safety_content, check_llm_or_exit,
)
from gemini_transcribe import transcribe_audio_gemini, gemini_transcription_configured

# ================================================================
#                      Environment & Setup
# ================================================================
load_dotenv(override=True)  # Load .env BEFORE using env vars, override OS environment

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

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, # Changed default to INFO
                    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')

# Suppress overly verbose logs from underlying libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("feedparser").setLevel(logging.INFO) # Allow feedparser info logs
disable_unreachable_local_proxies()


# --- Essential Environment Variables ---
DATABASE_URL = os.environ.get("AI_SAFETY_FEED_DB_URL")
if not DATABASE_URL:
    logging.critical("CRITICAL ERROR: AI_SAFETY_FEED_DB_URL environment variable not set. Cannot connect to database.")
    sys.exit(1)
check_llm_or_exit(embeddings=True)  # env check + one preflight call; aborts on a bad deployment/key

# Transcription (Gemini) is optional: without GEMINI_API_KEY we fall back to show notes.
transcription_configured = gemini_transcription_configured()
if not transcription_configured:
    logging.warning("GEMINI_API_KEY not set: audio transcription disabled, show notes will be used instead.")

# ================================================================
#    Relevance gate (shared implementation in llm_common)
# ================================================================

def is_ai_safety_post(title: str, html_body: str) -> Optional[bool]:
    """
    Fast yes/no guard-rail on the episode title + show notes.
    Returns True/False from the model, or None when the model could not be
    reached (the caller must then neither ingest nor record a skip).
    """
    text_content = ""
    if html_body:
        try:
            text_content = BeautifulSoup(html_body, 'html.parser').get_text(separator=' ', strip=True)
        except Exception as e:
            logging.warning(f"Failed to extract text from HTML: {e}")
            text_content = html_body
    return is_ai_safety_content(title, text_content[:1500], kind="podcast episode")

# ================================================================
#                         Constants
# ================================================================


# Ingest only posts published on/after this date (UTC, inclusive)
CUTOFF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
# Successful insertions per feed per run (a cost cap; raise via env once Gemini transcription cost is acceptable)
TARGET_INSERTIONS_PER_FEED = int(os.environ.get("PODCAST_MAX_INSERTS_PER_FEED", "2"))
# Process only the first N feeds (canary runs); None = all
MAX_FEEDS = None
# Safety limit: Max posts to *check* per feed if target not reached
MAX_POSTS_PER_FEED = 100

# --- Source Feeds ---
SOURCE_FEEDS = {
    "https://feeds.fame.so/ai-government-and-the-future": "AI, Government, and the Future",
    "https://feeds.feedburner.com/80000HoursPodcast": "80,000 Hours Podcast",
    "https://api.substack.com/feed/podcast/69345.rss": "Dwarkesh Podcast",
    "https://axrp.net/feed.xml": "AXRP",
    "https://anchor.fm/s/1e4a0eac/podcast/rss": "Machine Learning Street Talk",
    "https://futureoflife.org/podcast/feed/": "FLI Podcast",
    # NOTE: subscriber feed; currently returns HTML (token expired?). Refresh the uid from your account.
    "https://samharris.org/subscriber-rss/?uid=PNIypzHCbq78upz": "Making Sense with Sam Harris",
    "https://lexfridman.com/feed/podcast/": "Lex Fridman Podcast",
    "https://anchor.fm/s/ebbe3a98/podcast/rss": "For Humanity: An AI Safety Podcast",
    "https://podcast.clearerthinking.org/rss.xml": "Clearer Thinking",
    "http://feeds.libsyn.com/182816/rss": "Alignment Newsletter Podcast",
    "https://www.machine-ethics.net/itunes-rss-feed/": "Machine Ethics Podcast",
    "https://feeds.transistor.fm/intoaisafety": "Into AI Safety",
    # Center for AI Policy Podcast: official feed (feed.podbean.com/aipolicyus/feed.xml) returns 410 Gone
    # as of 2026-09; the show appears discontinued. Re-add if it resurfaces.
    "https://pinecast.com/feed/hear-this-idea": "Hear This Idea",
    "https://feeds.libsyn.com/539322/rss": "AI Governance Podcast",
    "https://feeds.acast.com/public/shows/5f2827aa17f940498f691817": "TechTank Podcast",
    "https://feeds.simplecast.com/rZ0cYk12": "Your Undivided Attention",
    "https://feeds.megaphone.fm/RINTP3108857801": "Cognitive Revolution",
}

# ================================================================
#                      Database Configuration
# ================================================================
# Define the columns in the 'content' table that we will insert into
# IMPORTANT: Ensure this order matches the data tuple created later.
# 'title_norm' IS included here, assuming it's a regular column to be inserted.
# If it's a DB-generated column, remove it from DB_COLS and the INSERT statement.
DB_COLS = (
    "source_url", "title", "source_type", "authors", "published_date",
    "topics", "score", "image_url", "sentence_summary", "paragraph_summary",
    "key_implication", "full_content", "full_content_markdown",
    "comment_count", "cluster_tag",
    "embedding_short", "embedding_full", 
    "audio_url",
    "cleaned_title",
)
NUM_DB_COLS = len(DB_COLS)

# Pre-compute the INSERT SQL statement for efficiency
# Uses ON CONFLICT with the 'title_norm' column to prevent duplicates
# NOTE: 'title_norm' is still the target for ON CONFLICT, even though it's not in the INSERT list.
INSERT_SQL = f"""
INSERT INTO content ({', '.join(DB_COLS)})
VALUES ({', '.join(['%s'] * NUM_DB_COLS)})
ON CONFLICT (title_norm) DO NOTHING;
"""

SKIP_INSERT_SQL = """
INSERT INTO skipped_posts (post_id, title_norm, source_url)
VALUES (%s, %s, %s)
ON CONFLICT DO NOTHING;
"""

def record_skip(cur, post_id: str, title_norm: str, source_url: Optional[str]):
    """Insert one row into skipped_posts (no commit)."""
    cur.execute(SKIP_INSERT_SQL, (post_id, title_norm, source_url))

# ================================================================
#                          Utility Helpers
# ================================================================

def normalise_title(title: str) -> str:
    """Normalizes a title string to match the database `title_norm` generation.

    This involves:
    1. Lowercasing the title.
    2. Replacing one or more whitespace characters with a single space.
    3. Stripping leading/trailing whitespace.
    """
    if not title: return ""
    # 1. Lowercase
    normalized = title.lower()
    # 2. Replace multiple whitespace chars with a single space
    normalized = re.sub(r'\s+', ' ', normalized)
    # 3. Strip leading/trailing whitespace
    normalized = normalized.strip()
    return normalized

def clean_title_for_storage(title: Optional[str]) -> Optional[str]:
    """Create a stable display-safe title variant for the cleaned_title column."""
    if not title:
        return None
    return re.sub(r'\s+', ' ', title).strip()

def iso_to_dt(iso_string: Optional[str]) -> Optional[datetime]:
    """Converts ISO 8601 string to timezone-aware datetime object (UTC)."""
    if not iso_string: return None
    try:
        dt_obj = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        if dt_obj.tzinfo is None:
            return dt_obj.replace(tzinfo=timezone.utc)
        return dt_obj.astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse ISO date string: '{iso_string}'. Error: {e}")
        return None

def struct_time_to_dt(st: Optional[time.struct_time]) -> Optional[datetime]:
    """Converts feedparser's time.struct_time to timezone-aware datetime (UTC)."""
    if not st: return None
    try:
        # Assume UTC if feedparser doesn't provide timezone info (common)
        dt_naive = datetime(*st[:6])
        return dt_naive.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, IndexError) as e:
        logging.warning(f"Could not convert time.struct_time {st} to datetime: {e}")
        return None

def safe_int_or_none(value: Any) -> Optional[int]:
    """Safely converts value to int, returning None on failure."""
    if value is None: return None
    try: return int(value)
    except (ValueError, TypeError): return None

def get_hostname_from_url(url: str) -> Optional[str]:
    """Extracts the hostname from a URL."""
    if not url: return None
    try: return urlparse(url).netloc.lower()
    except Exception as e:
        logging.warning(f"Could not parse hostname from URL '{url}': {e}")
        return None

# ================================================================
#                      Feed Fetching Iterator
# ================================================================

def extract_audio_url(entry):
    """Extract audio URL from RSS entry."""
    # Check enclosures
    for enc in entry.get("enclosures", []):
        if isinstance(enc, dict) and enc.get("type", "").startswith("audio/") and enc.get("href"):
            return enc["href"]
    
    # Check links
    for link in entry.get("links", []):
        if isinstance(link, dict) and link.get('rel') == 'enclosure' and link.get('type', '').startswith('audio/') and link.get('href'):
            return link["href"]
    
    # Check media_content
    for media in entry.get("media_content", []):
        if isinstance(media, dict) and media.get("medium") == "audio" and media.get("url"):
            return media["url"]
    
    # Check if main link is audio file
    link_url = entry.get("link", "")
    if link_url and link_url.endswith((".mp3", ".m4a", ".wav")):
        return link_url
    
    return None

def iter_rss_feed(feed_url: str, source_name: str) -> Dict[str, Any]:
    """
    Simplified iterator: Fetches and yields basic info for each entry in an RSS feed.
    Handles feed parsing errors gracefully.
    """
    logging.info(f"Fetching RSS feed: {source_name} ({feed_url})")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }
    feed_data = None
    try:
        # Consider adding timeout via requests if feedparser hangs
        feed_data = feedparser.parse(feed_url, request_headers=headers, agent=headers["User-Agent"])
        if feed_data.bozo:
             logging.warning(f"Feed '{source_name}' is not well-formed: {feed_data.bozo_exception}")
             # Continue processing if entries exist despite bozo flag

    except Exception as e:
         logging.error(f"Fatal error fetching or parsing feed '{source_name}' ({feed_url}): {e}", exc_info=True)
         return # Stop iteration for this feed on fatal error

    if not feed_data or not feed_data.entries:
        status_code = feed_data.get('status', 'N/A') if feed_data else 'N/A'
        logging.warning(f"RSS feed for '{source_name}' is empty or could not be parsed fully. Status: {status_code}. Contains {len(feed_data.get('entries', []))} entries.")
        return # Stop iteration if no entries

    logging.info(f"Found {len(feed_data.entries)} entries in feed '{source_name}'.")
    feed_title = (feed_data.feed.get("title") or "").strip() if getattr(feed_data, "feed", None) else ""
    significant = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", source_name)]
    if feed_title and significant and not any(w in feed_title.lower() for w in significant):
        logging.warning(f"FEED MISMATCH: configured '{source_name}' but the feed calls itself '{feed_title}' ({feed_url}). The feed may have moved.")

    for i, entry in enumerate(feed_data.entries):
        # --- Basic Data Extraction ---
        title = entry.get("title", "").strip()
        link = (entry.get("link") or "").strip()
        audio_url = extract_audio_url(entry)
        if not link:
            # Some feeds have no per-episode web page; fall back to the audio URL or GUID
            # so the episode is not silently dropped.
            link = (audio_url or entry.get("id") or "").strip()
        # Use link as fallback ID, ensure it's not empty
        post_id = entry.get("id", link) or f"{link}-{title}" # Create a more robust fallback ID

        if not title or not link or not post_id:
            logging.debug(f"Skipping entry {i+1} from '{source_name}': Missing essential field (title, link, or ID). Title='{title[:50]}...', Link='{link}', ID='{post_id}'")
            continue

        # --- Extract Content (HTML) ---
        html_body = "" 

        # Attempt 1: From entry.content
        content_items = entry.get("content")
        parsed_content_from_list = None # Stores result from this block
        if content_items and isinstance(content_items, list) and len(content_items) > 0:
            # Attempt 1a: specific 'text/html' type
            # Ensure 'item' is a dict before calling .get()
            val_html_type = next(
                (item.get('value', '') for item in content_items if isinstance(item, dict) and item.get('type') == 'text/html'),
                None 
            )
            
            # Attempt 1b: first item in list (this is the fallback value)
            val_first_item = "" # Default if first item is not a dict or has no value
            first_item_obj = content_items[0] # Known to exist due to len check
            if isinstance(first_item_obj, dict):
                val_first_item = first_item_obj.get('value', '')
            
            # Mimic original "A or B": if val_html_type is truthy, use it, else use val_first_item
            # This means if val_html_type is None or "", val_first_item will be used.
            parsed_content_from_list = val_html_type if val_html_type else val_first_item
        
        if parsed_content_from_list: # If content was found and is truthy from the list
            html_body = parsed_content_from_list
        else:
            # Attempt 2: From entry.summary_detail (if not found or empty from entry.content)
            summary_detail_obj = entry.get("summary_detail")
            val_summary_detail = None # Initialize
            if summary_detail_obj and isinstance(summary_detail_obj, dict):
                val_summary_detail = summary_detail_obj.get('value') # .get defaults to None, or returns actual value

            if val_summary_detail: # If val_summary_detail is truthy (not None, not empty string)
                html_body = val_summary_detail
            else:
                # Attempt 3: From entry.summary (if not found or empty from above)
                summary_val_direct = entry.get("summary") # This is typically the string itself or None
                # Ensure summary_val_direct is a string and truthy, as it might be other types from feedparser
                if summary_val_direct and isinstance(summary_val_direct, str): 
                    html_body = summary_val_direct
        
        # Ensure it's a string
        html_body = str(html_body) if html_body else ""

        # --- Extract Image URL ---
        image_url = None
        if entry.get("image") and isinstance(entry.image, dict) and entry.image.get("href"):
            image_url = entry.image.get("href")
        elif entry.get("media_content"):
             for media in entry.media_content:
                 if isinstance(media, dict) and media.get("medium") == "image" and media.get("url"):
                     image_url = media.get("url")
                     break
        elif entry.get("links"):
            for lnk in entry.links:
                if isinstance(lnk, dict) and lnk.get('rel') == 'enclosure' and lnk.get('type', '').startswith('image/') and lnk.get('href'):
                    image_url = lnk.get('href')
                    break

        # --- Extract Authors ---
        author_name = entry.get("author", source_name) # Default to source name
        authors_list = [name.strip() for name in re.split(r'\s+(?:,|&|and)\s+', author_name)] if author_name else [source_name]

        # --- Extract Tags ---
        tag_names = []
        if "tags" in entry and isinstance(entry.tags, list):
            tag_names = [t.term for t in entry.tags if t and hasattr(t, 'term') and t.term]

        # --- Yield Parsed Data ---
        yield {
            "post_id": post_id,
            "title": title,
            "link": link,
            "published_parsed": entry.get("published_parsed"), # Keep as struct_time for now
            "html_body": html_body,
            "audio_url": audio_url, # Still yield audio_url
            "image_url": image_url,
            "authors_list": authors_list,
            "tags": tag_names,
            "source_name": source_name,
            # Add other potentially useful raw fields if needed
            "raw_entry": entry # Optional: include for debugging or future use
        }

# ================================================================
#                     Main Processing Logic
# ================================================================

# Helper to make entry data JSON serializable
def make_serializable(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, time.struct_time):
        # Convert struct_time to a datetime object then to ISO format
        try:
            dt = struct_time_to_dt(obj) # Use existing helper
            return dt.isoformat() if dt else None
        except:
            return None # Or represent as string if conversion fails
    # Handle feedparser's FeedParserDict by converting to regular dict
    if isinstance(obj, feedparser.FeedParserDict):
         return dict(obj)
    # Recursively handle lists and dictionaries
    if isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    # Add handling for other non-serializable types if encountered
    # For example, if raw_entry contains complex objects:
    # if isinstance(obj, YourCustomClass):
    #     return obj.to_dict() # Assuming a method exists
    try:
        # Attempt default serialization for basic types
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        # Fallback for types json doesn't know how to handle
        return repr(obj) # Represent as string

def main():
    start_time = time.time()
    print("================================================================")
    print(f"Starting Podcast Feed Ingestion Script at {datetime.now(timezone.utc)}")
    print("*** Transcription: Gemini (gemini_transcribe.py); show notes used as fallback. ***")
    print("================================================================")
    # MODIFIED: Add a comment about the performance trade-off
    print("INFO: This version inserts rows individually, which may be slower than batching.")
    logging.info("Running in single-row insert mode (potentially slower than batching).")

    # Directory to save episode samples
    SAMPLE_DIR = "episode_samples"
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    print(f"Saving sample episode data to '{SAMPLE_DIR}/' directory.")

    conn = None
    existing_title_norms: Set[str] = set()
    already_skipped: Set[str] = set()
    skipped_post_ids: Set[str] = set()
    total_processed_count = 0
    total_skipped_count = 0
    total_inserted_count = 0
    total_failed_analysis_count = 0
    # MODIFIED: Renamed counter for clarity
    total_db_insert_failures = 0
    total_gate_errors = 0
    total_transcription_failures = 0

    try:
        # -------- 1. Database Connection & Setup --------
        print("\n--- Connecting to Database ---")
        logging.info(f"Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False # Manual transaction control
        register_vector(conn) # <-- ADDED for pgvector
        print("Database connection successful.")
        logging.info("Database connection successful.")

        # -------- 2. Fetch Existing Titles --------
        with conn.cursor() as cur:
            print("Fetching existing normalized titles from database...")
            cur.execute("SELECT title_norm FROM content WHERE title_norm IS NOT NULL") # Ensure not null
            existing_title_norms = {row[0] for row in cur.fetchall() if row[0]} # Filter None just in case
            print(f"--> Found {len(existing_title_norms):,} existing titles in the database.")
            logging.info(f"Fetched {len(existing_title_norms)} existing titles.")

            print("Fetching already skipped titles and IDs from database...")
            # Cache BOTH keys: skipped_posts' primary key is post_id, so a retitled episode
            # must still be recognised (otherwise the gate re-runs and the insert collides).
            cur.execute("SELECT post_id, title_norm FROM skipped_posts")
            skipped_rows = cur.fetchall()
            already_skipped = {r[1] for r in skipped_rows if r[1]}
            skipped_post_ids = {r[0] for r in skipped_rows if r[0]}
            print(f"--> Found {len(skipped_rows):,} already skipped posts in the database.")
            logging.info(f"Fetched {len(skipped_rows)} already skipped posts.")

        # -------- 3. Process Feeds --------
        feeds_to_run = list(SOURCE_FEEDS.items())[:MAX_FEEDS] if MAX_FEEDS else list(SOURCE_FEEDS.items())
        print(f"\n--- Starting Processing for {len(feeds_to_run)} Feeds (max {TARGET_INSERTIONS_PER_FEED} inserts each) ---")
        feed_counter = 0
        for feed_url, source_name in feeds_to_run:
            feed_counter += 1
            print(f"\n===== [{feed_counter}/{len(feeds_to_run)}] Processing Feed: {source_name} =====")
            posts_in_feed_count = 0
            posts_processed_in_feed = 0
            posts_skipped_in_feed = 0
            sample_saved_for_feed = False # Flag to save only one sample per feed
            successful_insertions_in_feed = 0 # Counter for successful DB inserts this feed

            # Use the simplified iterator
            for entry_data in iter_rss_feed(feed_url, source_name):
                # MODIFIED: Moved sample saving to the very beginning for the first entry
                if not sample_saved_for_feed:
                    try:
                        title_for_log = entry_data.get('title', '[No Title]')[:50]
                        # Sanitize source_name for filename
                        safe_source_name = re.sub(r'[^\w\-]+', '_', source_name)
                        sample_filename = os.path.join(SAMPLE_DIR, f"{safe_source_name}_sample.json")

                        # Make a copy and serialize complex types
                        serializable_data = make_serializable(entry_data.copy())

                        with open(sample_filename, 'w', encoding='utf-8') as f:
                            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
                        logging.info(f"  -> Saved first entry sample for '{title_for_log}...' to {sample_filename}")
                        sample_saved_for_feed = True # Mark as saved for this feed
                    except Exception as sample_err:
                        logging.error(f"  ERROR: Failed to save first entry sample for '{title_for_log}...': {sample_err}", exc_info=True)
                    # --- End Save Sample Data ---

                total_processed_count += 1
                title = entry_data['title']
                link = entry_data['link']
                post_id = entry_data['post_id'] # For logging
                logging.info(f"--- Processing Entry #{posts_in_feed_count}: '{title[:70]}...' (ID: {post_id}) ---")

                # --- 3a. Normalize Title & Check Existence ---
                title_norm = normalise_title(title)
                # --- DEBUG: Print title and normalized title ---
                # print(f"  DEBUG CHECK: Original Title = '{title}'")
                # print(f"  DEBUG CHECK: Normalized Title = '{title_norm}'")
                # --- End DEBUG ---

                # Check if already processed or known to be skipped
                if title_norm in existing_title_norms or title_norm in already_skipped or post_id in skipped_post_ids:
                    logging.debug(f"  SKIP (silent): Normalized title '{title_norm[:70]}...' already in content or skipped_posts cache.")
                    # No counter increment for this type of skip, as it's a pre-existing state.
                    # posts_skipped_in_feed might still be relevant if we want to track how many were skipped *per feed* due to cache.
                    # For now, keeping it truly silent as per instruction for global counters.
                    continue

                # Count only entries that need work, so cached duplicates don't eat the limit.
                posts_in_feed_count += 1
                if posts_in_feed_count > MAX_POSTS_PER_FEED:
                     logging.warning(f"SAFETY BREAK: Reached max check limit ({MAX_POSTS_PER_FEED}) for feed '{source_name}' without reaching target insertions ({TARGET_INSERTIONS_PER_FEED}). Moving to next feed.")
                     break # Stop processing this feed

                if not title_norm:
                    logging.warning(f"  SKIP: Could not normalize title for post ID {post_id}. Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm); skipped_post_ids.add(post_id)
                        logging.debug(f"  Recorded skip for '{title_norm[:70]}...' (empty title_norm) and added to already_skipped cache.")
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for post ID {post_id} (empty title_norm). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    posts_skipped_in_feed += 1
                    continue

                # --- 3b. Parse Date & Check Cutoff ---
                published_dt = struct_time_to_dt(entry_data.get("published_parsed"))
                if not published_dt:
                    logging.warning(f"  SKIP: Could not parse publication date for post ID {post_id}. Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm); skipped_post_ids.add(post_id)
                        logging.debug(f"  Recorded skip for '{title_norm[:70]}...' (invalid date) and added to already_skipped cache.")
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for post '{title_norm[:70]}...' (invalid date). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    posts_skipped_in_feed += 1
                    continue
                if published_dt < CUTOFF_DATE:
                    logging.info(f"  SKIP: Post published ({published_dt.date()}) before cutoff date ({CUTOFF_DATE.date()}). Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm); skipped_post_ids.add(post_id)
                        logging.debug(f"  Recorded skip for '{title_norm[:70]}...' (before cutoff) and added to already_skipped cache.")
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for post '{title_norm[:70]}...' (before cutoff). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    posts_skipped_in_feed += 1
                    continue

                # --- 3c. AI Safety Guardrail ---
                html_body = entry_data.get('html_body', '')
                gate_verdict = is_ai_safety_post(title, html_body)
                if gate_verdict is None:
                    logging.warning(f"  RETRY LATER: Guard-rail unavailable for '{title[:70]}...'. Not recorded; will retry next run.")
                    total_gate_errors += 1
                    continue
                if not gate_verdict:
                    logging.info(f"  SKIP: Post '{title[:70]}...' failed AI safety guardrail check. Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm); skipped_post_ids.add(post_id)
                        logging.debug(f"  Recorded skip for '{title_norm[:70]}...' (guardrail fail) and added to already_skipped cache.")
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for post '{title_norm[:70]}...' (guardrail fail). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    posts_skipped_in_feed += 1
                    continue

                # --- If passed checks, proceed with full processing ---
                posts_processed_in_feed += 1

                # --- Initialize content variables ---
                # MODIFIED: Clearer variable names
                transcript_text = None
                html_show_notes = html_body # Keep original HTML if needed
                markdown_from_notes = ""
                text_from_notes = ""
                final_analysis_content = "" # This will hold the content used for AI analysis
                final_full_content_raw = "" # This will be saved to DB 'full_content'
                final_full_content_markdown = "" # This will be saved to DB 'full_content_markdown'
                audio_url = entry_data.get('audio_url') # Get audio URL

                # --- 3d. Attempt Transcription with Gemini ---
                transcription_failed = False
                if transcription_configured and audio_url:
                    tr = transcribe_audio_gemini(audio_url, title)
                    transcript_text = tr.text if tr.ok else None
                    transcription_failed = (tr.status == "failed")
                    if transcription_failed:
                        logging.warning(f"  Transcription failed ({tr.reason}); episode will be retried next run.")
                elif not audio_url:
                     logging.info("  Skipping transcription: No audio URL found in feed entry.")

                if transcription_failed:
                    # A transient failure must not become a notes-only row or a permanent skip.
                    total_transcription_failures += 1
                    continue

                # --- 3e. Prepare Content for Analysis ---
                if transcript_text:
                    logging.info("  Using Gemini transcript for content analysis.")
                    final_analysis_content = transcript_text
                    final_full_content_raw = transcript_text
                    # Use the raw transcript as markdown for now.
                    # Could potentially add markdown formatting later if needed.
                    final_full_content_markdown = transcript_text
                elif html_show_notes:
                    logging.info("  Transcription failed or skipped. Falling back to HTML show notes for analysis.")
                    try:
                        soup = BeautifulSoup(html_show_notes, 'html.parser')
                        # Existing HTML cleaning logic
                        for element in soup(["script", "style", "iframe", "form", "button", "input", "noscript", "header", "footer", "nav", "aside"]):
                            element.decompose()
                        cleaned_html = str(soup)
                        logging.debug("HTML cleaning successful.")
                        try:
                            # Use markdownify result for both markdown and raw text
                            md_from_html = markdownify(cleaned_html, heading_style="ATX", bullets="-", strip=['script', 'style'], escape_underscores=False)                             
                            markdown_from_notes = md_from_html
                            # Extract plain text from markdown/html for raw content
                            soup_text = BeautifulSoup(md_from_html, 'html.parser') # Parse the markdown
                            text_from_notes = soup_text.get_text(separator=' ', strip=True)

                            final_analysis_content = text_from_notes # Use text from notes for analysis
                            final_full_content_raw = text_from_notes
                            final_full_content_markdown = markdown_from_notes
                            logging.debug("Markdown conversion and text extraction from HTML successful.")
                        except Exception as md_e:
                            logging.error(f"  ERROR: Markdownify conversion/text extraction failed: {md_e}", exc_info=True)
                            # No usable content from HTML if markdownify fails badly
                    except Exception as bs_e:
                        logging.error(f"  ERROR: BeautifulSoup cleaning failed: {bs_e}", exc_info=True)
                        # No usable content from HTML if cleaning fails
                else:
                    logging.warning(f"  No transcript and no HTML show notes found for post ID {post_id}. Cannot perform analysis.")
                    # Ensure content variables are empty/None
                    final_analysis_content = ""
                    final_full_content_raw = ""
                    final_full_content_markdown = ""

                # --- Check for usable content before AI analysis ---
                analysis_content = final_analysis_content # This is from line 1041 in original file
                if not (analysis_content and analysis_content.strip()):
                    logging.warning(f"  SKIP: Post '{title[:70]}...' has no usable content (transcript/HTML). Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm); skipped_post_ids.add(post_id)
                        logging.debug(f"  Recorded skip for '{title_norm[:70]}...' (no usable content) and added to already_skipped cache.")
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for post '{title_norm[:70]}...' (no usable content). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    posts_skipped_in_feed += 1
                    continue # Skip AI analysis and DB insertion

                # --- 3f. Perform AI Analyses (one structured call, see llm_common) ---
                # A failure means the episode is NOT inserted. Transient errors leave it
                # unrecorded so the next run retries; content-filter rejections are recorded.
                logging.info("  -> Performing AI analysis on available content...")
                try:
                    analysis = analyze_content(title, analysis_content, entry_data.get('tags', []))
                except LLMContentFiltered as e:
                    logging.warning(f"  SKIP: Content filter blocked '{title[:70]}...': {e}. Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm); skipped_post_ids.add(post_id)
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for '{title_norm[:70]}...' (content filter). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    total_failed_analysis_count += 1
                    continue
                except (LLMError, ValueError) as e:
                    logging.error(f"  RETRY LATER: Analysis failed for '{title[:70]}...': {e}. Not inserted.")
                    total_failed_analysis_count += 1
                    continue

                sentence_summary = analysis["sentence_summary"]
                paragraph_summary = analysis["paragraph_summary"]
                key_implication = analysis["key_implication"]
                db_cluster = analysis["cluster_tag"]
                db_tags = analysis["tags"]
                logging.debug(f"     Cluster='{db_cluster}', Tags={db_tags}")

                # OpenAI Embeddings
                logging.info("  -> Generating Embeddings...")
                embedding_short_vector, embedding_full_vector = generate_embeddings(
                    title or "",
                    build_embedding_text(sentence_summary, paragraph_summary, key_implication, db_tags),
                )
                if embedding_short_vector is None or embedding_full_vector is None:
                    logging.error(f"  RETRY LATER: Embedding generation failed for '{title[:70]}...'. Not inserted.")
                    total_failed_analysis_count += 1
                    continue

                # --- 3g. Prepare Data Tuple for Insertion ---
                # Ensure order matches DB_COLS exactly!
                # 'full_content' is raw text from notes, 'full_content_markdown' is markdown from notes.
                cleaned_title = None  # left NULL on insert; rewrite_titles.py --mode titles fills it (frontend/backend fall back to title)
                data_tuple = (
                    link,                           # source_url
                    title,                          # title
                    entry_data['source_name'],      # source_type
                    entry_data.get('authors_list', ['Unknown']), # authors (list for ARRAY type)
                    published_dt,                   # published_date (datetime or None)
                    db_tags,                        # topics (list[str] or None)
                    0,                              # score (Podcasts do not provide score; default to 0)
                    entry_data.get('image_url'),    # image_url (str or None)
                    sentence_summary,               # sentence_summary (str or None)
                    paragraph_summary,              # paragraph_summary (str or None)
                    key_implication,                # key_implication (str or None)
                    final_full_content_raw or None, # full_content (Transcript or text from notes)
                    final_full_content_markdown or None, # full_content_markdown (Transcript or markdown notes)
                    None,                           # comment_count (Podcasts don't usually have comments) - Use None
                    db_cluster,                     # cluster_tag (str or None)
                    embedding_short_vector,         # embedding_short (list[float] or None)
                    embedding_full_vector,          # embedding_full (list[float] or None)
                    audio_url,                      # audio_url (original URL, kept for reference)
                    cleaned_title,                  # cleaned_title (str or None)
                    # title_norm                      # Removed: Generated by the database
                )

                # --- Data Validation before adding to batch ---
                if len(data_tuple) != NUM_DB_COLS:
                     logging.critical(f"FATAL MISMATCH: Data tuple length ({len(data_tuple)}) != NUM_DB_COLS ({NUM_DB_COLS}) for post '{title[:50]}...'. Skipping insertion. Check DB_COLS definition.")
                     continue # Skip this post

                # ========================================================
                # MODIFIED: Insert this single row immediately
                # ========================================================
                logging.debug(f"Attempting to insert post '{title[:50]}...'")
                try:
                    with conn.cursor() as cur:
                        # INSERT_SQL handles ON CONFLICT DO NOTHING
                        cur.execute(INSERT_SQL, data_tuple)
                        conn.commit() # Commit this single row transaction
                        logging.info(f"  SUCCESS: DB insert processed for '{title[:70]}...' (inserted or ignored by ON CONFLICT).")
                        total_inserted_count += 1 # Count attempts that didn't raise DB error
                        # Add to set to prevent reprocessing duplicates later in this run
                        # even if ON CONFLICT ignored the insert.
                        existing_title_norms.add(title_norm)
                        successful_insertions_in_feed += 1 # Increment feed-specific counter
                except (psycopg2.DatabaseError, Psycopg2OpError) as db_err:
                    logging.error(f"  DB ERROR: Failed to insert post '{title[:70]}...' (ID: {post_id}). Error: {db_err}", exc_info=False) # Keep log concise
                    logging.debug(f"Failed data tuple: {data_tuple}", exc_info=True) # Add full trace in debug
                    conn.rollback() # Rollback the failed transaction for this row
                    total_db_insert_failures += 1
                except Exception as e:
                    logging.error(f"  UNEXPECTED ERROR: Failed to insert post '{title[:70]}...' (ID: {post_id}). Error: {e}", exc_info=True)
                    conn.rollback() # Rollback on unexpected errors
                    total_db_insert_failures += 1
                # ========================================================
                # End of single row insertion block
                # ========================================================

                # --- Check if target insertions reached ---
                if successful_insertions_in_feed >= TARGET_INSERTIONS_PER_FEED:
                    logging.info(f"Reached target insertions ({TARGET_INSERTIONS_PER_FEED}) for feed '{source_name}'. Moving to next feed.")
                    break # Stop processing this feed

            logging.info(f"===== Finished Feed: {source_name} =====")
            logging.info(f"    Entries checked: {posts_in_feed_count}")
            logging.info(f"    Entries processed (passed checks): {posts_processed_in_feed}")
            logging.info(f"    Entries skipped (duplicate/date/guardrail): {posts_skipped_in_feed}")
            time.sleep(1) # Be polite between feeds

    except Psycopg2OpError as e:
        logging.critical(f"FATAL: Database connection failed: {e}", exc_info=True)
        print(f"FATAL ERROR: Database connection failed: {e}")
    except KeyboardInterrupt:
         logging.warning("KeyboardInterrupt received. Attempting graceful shutdown.")
         print("\nKeyboard interrupt detected. Shutting down...")
         if conn: conn.rollback() # Rollback any pending transaction
    except Exception as e:
        logging.critical(f"CRITICAL ERROR in main processing loop: {e}", exc_info=True)
        if conn: conn.rollback()
        sys.exit(1)
    finally:
        if conn:
            conn.close()
        print("\n--- Podcast Ingestion Summary ---")
        print(f"Entries examined:                 {total_processed_count}")
        print(f"Rows inserted (or already there): {total_inserted_count}")
        print(f"Skipped and recorded:             {total_skipped_count}")
        print(f"Analysis/embedding failures:      {total_failed_analysis_count}")
        print(f"Gate unavailable (retry later):   {total_gate_errors}")
        print(f"Transcription failed (retry later): {total_transcription_failures}")
        print(f"DB insert failures:               {total_db_insert_failures}")
        print(f"Elapsed:                          {time.time() - start_time:.0f}s")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Podcast RSS ingestion")
    ap.add_argument("--max-feeds", type=int, default=None, help="Process only the first N feeds (canary runs)")
    ap.add_argument("--max-inserts-per-feed", type=int, default=None, help="Override PODCAST_MAX_INSERTS_PER_FEED")
    _args = ap.parse_args()
    if _args.max_inserts_per_feed:
        TARGET_INSERTIONS_PER_FEED = _args.max_inserts_per_feed
    MAX_FEEDS = _args.max_feeds
    main()
