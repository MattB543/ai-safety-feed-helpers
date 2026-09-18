#!/usr/bin/env python3
"""
Ingest top‑quality AI‑safety content from EA Forum, LessWrong, and Alignment Forum
into the `content` table of the AI‑Safety‑Feed database.

This script fetches posts via GraphQL, filters them based on date, tags, score,
and comment counts, performs analysis (summarization, implication identification,
clustering/tagging) using Azure OpenAI chat completions, and inserts the processed data into
a PostgreSQL database, handling potential duplicates based on normalized titles.

All Substack‑specific logic has been removed.
"""

# ================================================================
#                            Imports
# ================================================================
import os
import re
import json
import time
import logging
from datetime import datetime, timezone, date # Explicit imports for clarity
import sys # For exiting early

# Fix Windows console encoding issues with Unicode characters - must be done early
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from markdownify import markdownify
from bs4 import BeautifulSoup
import psycopg2
from psycopg2 import extras # Explicit import for batch insertion
from pgvector.psycopg2 import register_vector # <<< ADD THIS IMPORT
from dotenv import load_dotenv

# ================================================================
#                      Environment & Setup
# ================================================================
load_dotenv(override=True)  # Load .env BEFORE using env vars, override OS environment

# Shared Azure/OpenAI helpers: one structured analysis call per post, embeddings, retries.
from llm_common import (
    LLMContentFiltered, LLMError, analyze_content, build_embedding_text,
    generate_embeddings, check_llm_or_exit,
)

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

# --- Essential Environment Variables ---
DATABASE_URL = os.environ.get("AI_SAFETY_FEED_DB_URL")

if not DATABASE_URL:
    print("CRITICAL ERROR: AI_SAFETY_FEED_DB_URL environment variable not set. Cannot connect to database.")
    sys.exit(1) # Exit if DB URL is missing

check_llm_or_exit(embeddings=True)  # env check + one preflight call; aborts on a bad deployment/key

# --- Logging Configuration (Optional but recommended) ---
# Basic logging setup - consider more advanced config for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.WARNING) # Added for requests/urllib3 noise
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING) # Suppress OpenAI logs if needed
disable_unreachable_local_proxies()

# ================================================================
#                     Forum‑specific constants
# ================================================================
EA_API_URL = "https://forum.effectivealtruism.org/graphql"
EA_AI_SAFETY_TAG_ID = "oNiQsBHA3i837sySD"

LW_API_URL = "https://www.lesswrong.com/graphql"
LW_AI_SAFETY_TAG_ID = "yBXKqk8wEg6eM8w5y"


DEFAULT_LIMIT = 3000 # Max posts to fetch per source initially
BATCH_SIZE    = 3   # Rows per database INSERT batch

# Ingest only posts published on/after this date (UTC, inclusive)
CUTOFF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)

# ================================================================
#                 API Rate Limiting Settings
# ================================================================
# Delay between GraphQL API calls to avoid rate limiting
API_CALL_DELAY_SEC = 2

# ================================================================
#                 Filtering Thresholds & Tag Sets
# ================================================================
# --- Score/Comment Thresholds ---
# EA Forum
EA_SCORE_THRESHOLD_HIGH         = 85
EA_COMMENT_THRESHOLD_HIGH_SCORE = 0
EA_SCORE_THRESHOLD_MID          = 65
EA_COMMENT_THRESHOLD_MID_SCORE  = 12

# LessWrong
LW_SCORE_THRESHOLD_HIGH         = 85
LW_COMMENT_THRESHOLD_HIGH_SCORE = 0
LW_SCORE_THRESHOLD_MID          = 65
LW_COMMENT_THRESHOLD_MID_SCORE  = 20

# --- Tag Sets for Filtering ---
APRIL_FOOLS_TAGS = {"April Fool's", "April Fools' Day"} # Set for fast lookups
AI_TAGS_LW       = {"AI"} # Required tag for LW posts

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
    "embedding_short",
    "embedding_full",
    "cleaned_title",
)
NUM_DB_COLS = len(DB_COLS) # Calculate number of placeholders needed

# Pre-compute the INSERT SQL statement for efficiency
INSERT_SQL = f"""
INSERT INTO content ({', '.join(DB_COLS)})
VALUES ({', '.join(['%s'] * NUM_DB_COLS)})
ON CONFLICT (title_norm) DO NOTHING;
"""

# SQL and helper for recording skipped posts
SKIP_INSERT_SQL = """
INSERT INTO skipped_posts (post_id, title_norm, source_url)
VALUES (%s, %s, %s)
ON CONFLICT DO NOTHING;
"""

def record_skip(cur, post_id: str, title_norm: str, source_url: str | None):
    """Insert one row into skipped_posts (no commit)."""
    # Ensure post_id and source_url are strings, handle None for source_url
    cur.execute(SKIP_INSERT_SQL, (
        str(post_id) if post_id is not None else 'N/A',
        str(title_norm),
        str(source_url) if source_url is not None else None
    ))

# ================================================================
#                         Utility Helpers
# ================================================================

def normalise_title(title: str) -> str:
    """
    Normalizes a title string for consistent comparison and database indexing.
    Converts to lowercase, replaces multiple whitespace chars with a single space,
    and strips leading/trailing whitespace.

    Args:
        title: The original title string.

    Returns:
        The normalized title string.
    """
    if not title: return ""
    return re.sub(r"\s+", " ", title).strip().lower()

def clean_title_for_storage(title: str | None) -> str | None:
    """Create a stable display-safe title variant for the cleaned_title column."""
    if not title:
        return None
    return re.sub(r"\s+", " ", title).strip()

def iso_to_dt(iso_string: str | None) -> datetime | None:
    """
    Safely converts an ISO 8601 timestamp string to a timezone-aware
    datetime object (UTC). Handles 'Z' notation and naive datetimes.

    Args:
        iso_string: The ISO 8601 formatted string.

    Returns:
        A timezone-aware datetime object (UTC) or None if parsing fails.
    """
    if not iso_string:
        return None
    try:
        # Handle 'Z' suffix for UTC and ensure timezone awareness
        dt_obj = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        if dt_obj.tzinfo is None:
            # Assume UTC if timezone is naive
            return dt_obj.replace(tzinfo=timezone.utc)
        # Convert to UTC if it has another timezone
        return dt_obj.astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse ISO date string: '{iso_string}'. Error: {e}")
        return None

def safe_int_or_zero(value: any) -> int:
    """
    Safely converts a value to an integer. Returns 0 if the value is None,
    cannot be converted, or causes a TypeError/ValueError.

    Args:
        value: The value to convert.

    Returns:
        The integer representation or 0.
    """
    if value is None: return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def safe_join(separator: str, items: list[str | None]) -> str:
    """
    Joins a list of strings or None with a separator, skipping None or empty/whitespace items.
    """
    return separator.join(item for item in items if item and isinstance(item, str) and item.strip())

# ================================================================
#                      GraphQL Fetching Logic
# ================================================================

# Define the fields we want to retrieve for each post
POST_FIELDS = """
  _id           # Unique identifier for the post
  title         # Post title
  pageUrl       # Canonical URL of the post
  commentCount  # Number of comments
  baseScore     # Score/karma of the post
  af            # True when the post is also on the Alignment Forum
  postedAt      # Publication timestamp (ISO 8601)
  htmlBody      # Full HTML content of the post
  tags {        # Associated tags
    _id
    name
  }
  user {        # Primary author information
    displayName
  }
  coauthors {   # Co-author information (if any)
    displayName
  }
"""

def get_forum_posts(api_url: str, tag_id: str | None = None, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """
    Queries a forum's GraphQL API for posts.

    Handles fetching posts either by a specific tag ID (for EA/LW) or
    the default 'top' view (for AF). Includes error handling and logging.

    Args:
        api_url: The GraphQL endpoint URL for the forum.
        tag_id: The specific tag ID to filter by (optional).
        limit: The maximum number of posts to retrieve.

    Returns:
        A list of post dictionaries, or an empty list if fetching fails or
        no posts are found.
    """
    # Determine the view clause based on whether a tag_id is provided
    if tag_id:
        view_clause = f'view: "tagById", tagId: "{tag_id}"'
        source_desc = f"{api_url} (Tag ID: {tag_id})"
    else:
        view_clause = 'view: "top"' # Default view for forums like AF
        source_desc = f"{api_url} (View: top)"

    # Construct the GraphQL query using an f-string
    query = f"""
    {{
      posts(
        input: {{
          terms: {{
            {view_clause}
            limit: {limit}
            # Add other terms like 'sort', 'before', 'after' if needed later
          }}
        }}
      ) {{
        results {{
          {POST_FIELDS}
        }}
      }}
    }}
    """

    # Standard headers for the request
    headers = {
        "Content-Type": "application/json",
        # Use a more descriptive User-Agent
        "User-Agent": "AISafetyFeed/1.0 (+https://aisafetyfeed.com)"
    }

    print(f"Executing GraphQL query for {limit} posts from {source_desc}...")
    logging.info(f"Executing GraphQL query for {limit} posts from {source_desc}")

    # Retry logic for rate limiting with exception handling
    max_retries = 4
    retry_delays = [10, 30, 60]  # Backoff delays in seconds

    try:
        for attempt in range(max_retries):
            try:
                response = requests.post(api_url, json={"query": query}, headers=headers, timeout=90)

                # Handle rate limiting with retry
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        print(f"  Rate limit hit (429). Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        logging.warning(f"Rate limit (429) for {source_desc}. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        # Final attempt failed
                        logging.error(f"Query failed for {source_desc}: 429 Client Error: Too Many Requests after {max_retries} attempts")
                        print(f"ERROR: Query failed for {source_desc}: Rate limit (429) persisted after {max_retries} attempts.")
                        return []

                response.raise_for_status() # Raise HTTPError for other bad responses (4xx or 5xx)

                result = response.json()

                # Check for GraphQL-specific errors returned in the response body
                if "errors" in result:
                    logging.error(f"GraphQL API ({source_desc}) returned errors: {json.dumps(result['errors'], indent=2)}")
                    print(f"ERROR: GraphQL API ({source_desc}) returned errors. Check logs.")
                    # Optionally log the failed query for debugging (be careful with sensitive data if any)
                    # logging.debug(f"Failed GraphQL query was:\n{query}")
                    return [] # Return empty list on GraphQL errors

                # Check for expected data structure
                if "data" not in result or "posts" not in result["data"] or "results" not in result["data"]["posts"]:
                     logging.warning(f"Unexpected response structure from {source_desc}. 'data.posts.results' not found.")
                     print(f"WARNING: Unexpected response structure from {source_desc}. Check logs.")
                     # Log the actual data received for debugging
                     # logging.debug(f"Response data received: {json.dumps(result.get('data', {}), indent=2)}")
                     return [] # Return empty list if structure is wrong

                posts_data = result["data"]["posts"]["results"]
                print(f"Successfully fetched {len(posts_data)} posts from {source_desc}.")
                logging.info(f"Successfully fetched {len(posts_data)} posts from {source_desc}.")

                # Add delay between successful API calls to avoid rate limiting
                time.sleep(API_CALL_DELAY_SEC)
                return posts_data

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                # Transient transport failure (the forums sometimes drop the large 3000-post
                # response): retry with the same backoff used for 429s.
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    print(f"  Transient network error ({type(e).__name__}). Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    logging.warning(f"Transient network error for {source_desc}: {e}. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                logging.error(f"Query failed for {source_desc} after {max_retries} attempts: {e}")
                print(f"ERROR: Query failed for {source_desc} after {max_retries} attempts: {e}")
                return []
            except requests.exceptions.HTTPError as e:
                # Handle other HTTP errors (not 429, which is handled above)
                logging.error(f"Query failed for {source_desc}: {e}", exc_info=True)
                print(f"ERROR: Query failed for {source_desc}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logging.error(f"Response status code: {e.response.status_code}")
                return []

    except requests.exceptions.Timeout:
        logging.error(f"Query failed for {source_desc}: Request timed out.")
        print(f"ERROR: Query failed for {source_desc}: Request timed out.")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Query failed for {source_desc}: {e}", exc_info=True)
        print(f"ERROR: Query failed for {source_desc}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response status code: {e.response.status_code}")
            # Avoid logging potentially large response bodies directly unless debugging
            # logging.debug(f"Response body (truncated): {e.response.text[:500]}...")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response from {source_desc}: {e}")
        print(f"ERROR: Failed to decode JSON response from {source_desc}. Check logs.")
        # Log the response text that failed to parse
        # if hasattr(response, 'text'):
        #     logging.debug(f"Response text was (truncated): {response.text[:500]}...")
        return []
    except Exception as e:
        # Catch any other unexpected errors during the fetch process
        logging.error(f"An unexpected error occurred fetching posts from {source_desc}: {e}", exc_info=True)
        print(f"ERROR: An unexpected error occurred fetching posts from {source_desc}. Check logs.")
        return []

# ================================================================
#                     Source‑Specific Filtering
# ================================================================

def filter_ea_posts(posts: list[dict], tag_id: str) -> list[dict]:
    """Filters EA Forum posts based on date, AI safety tag, score/comments, and excludes April Fools'."""
    print(f"\n--- Filtering {len(posts)} EA Forum posts ---")
    if not posts: return []

    filtered_posts = []
    for p in posts:
        # Basic check for essential fields
        if not p or not p.get('pageUrl') or not p.get('postedAt'):
            logging.debug(f"Skipping EA post due to missing essential fields: {p.get('_id', 'N/A')}")
            continue

        # 1. Filter by date
        posted_at_dt = iso_to_dt(p.get("postedAt"))
        if not posted_at_dt or posted_at_dt < CUTOFF_DATE:
            continue # Skip if date is invalid or before cutoff

        # 2. Check tags: Must have AI safety tag, must NOT have April Fools' tag
        post_tags = p.get("tags", [])
        has_ai_safety_tag = any(t and t.get("_id") == tag_id for t in post_tags)
        has_april_fools_tag = any(t and t.get("name") in APRIL_FOOLS_TAGS for t in post_tags)

        if not has_ai_safety_tag or has_april_fools_tag:
            continue # Skip if wrong tags

        # 3. Apply score and comment filter using defined thresholds
        score = safe_int_or_zero(p.get("baseScore"))
        comments = safe_int_or_zero(p.get("commentCount"))

        passes_threshold = (
            (score >= EA_SCORE_THRESHOLD_HIGH and comments >= EA_COMMENT_THRESHOLD_HIGH_SCORE) or
            (score > EA_SCORE_THRESHOLD_MID and comments > EA_COMMENT_THRESHOLD_MID_SCORE)
        )

        if passes_threshold:
            p["source_type"] = "EA Forum" # Add source type identifier
            filtered_posts.append(p)

    print(f"--- Found {len(filtered_posts)} EA Forum posts meeting criteria ---")
    logging.info(f"Filtered EA Forum posts: {len(posts)} -> {len(filtered_posts)}")
    return filtered_posts

def filter_lw_posts(posts: list[dict]) -> list[dict]:
    """Filters LessWrong posts based on date, 'AI' tag, score/comments, and excludes April Fools'."""
    print(f"\n--- Filtering {len(posts)} LessWrong posts ---")
    if not posts: return []

    filtered_posts = []
    af_count = 0
    for p in posts:
        # Basic check
        if not p or not p.get('pageUrl') or not p.get('postedAt'):
            logging.debug(f"Skipping LW post due to missing essential fields: {p.get('_id', 'N/A')}")
            continue

        # 1. Filter by date
        posted_at_dt = iso_to_dt(p.get("postedAt"))
        if not posted_at_dt or posted_at_dt < CUTOFF_DATE:
            continue

        # 2. Check tags: Must include "AI", must NOT include April Fools'
        # Use set for efficient lookup
        tag_names = {t.get("name") for t in p.get("tags", []) if t and t.get("name")}
        has_ai_tag = bool(AI_TAGS_LW & tag_names) # Check intersection with required AI tags
        has_april_fools_tag = bool(APRIL_FOOLS_TAGS & tag_names) # Check intersection with April Fools tags

        if not has_ai_tag or has_april_fools_tag:
            continue

        # 3. Apply score and comment filter
        score = safe_int_or_zero(p.get("baseScore"))
        comments = safe_int_or_zero(p.get("commentCount"))

        passes_threshold = (
            (score >= LW_SCORE_THRESHOLD_HIGH and comments >= LW_COMMENT_THRESHOLD_HIGH_SCORE) or
            (score > LW_SCORE_THRESHOLD_MID and comments > LW_COMMENT_THRESHOLD_MID_SCORE)
        )

        if passes_threshold:
            if p.get("af"):
                # Alignment Forum posts are LessWrong posts flagged af=true. Label them as
                # AF and link to the AF copy instead of querying the AF endpoint separately.
                p["source_type"] = "Alignment Forum"
                p["pageUrl"] = (p["pageUrl"]
                                .replace("www.lesswrong.com", "www.alignmentforum.org")
                                .replace("lesswrong.com", "alignmentforum.org"))
                af_count += 1
            else:
                p["source_type"] = "Less Wrong" # Add source type
            filtered_posts.append(p)

    print(f"--- Found {len(filtered_posts)} LessWrong posts meeting criteria ({af_count} flagged Alignment Forum) ---")
    logging.info(f"Filtered LessWrong posts: {len(posts)} -> {len(filtered_posts)}")
    return filtered_posts

# ================================================================
#                     Deduplication Helper
# ================================================================

def choose_highest_score(posts: list[dict]) -> list[dict]:
    """
    Deduplicates a list of post dictionaries based on normalized titles.
    If multiple posts share the same normalized title, only the one with
    the highest score is kept. Posts without titles are discarded.

    Args:
        posts: A list of post dictionaries.

    Returns:
        A list of unique post dictionaries, keeping the highest-scoring duplicates.
    """
    print(f"\n--- Deduplicating {len(posts)} posts in memory by normalized title (keeping highest score) ---")
    posts_by_norm_title: dict[str, dict] = {}
    valid_post_count = 0
    discarded_no_title = 0

    for p in posts:
        title = p.get("title")
        if not title:
            discarded_no_title += 1
            continue # Cannot deduplicate without a title

        valid_post_count += 1
        norm_title = normalise_title(title)
        current_score = safe_int_or_zero(p.get("baseScore"))

        # Check if we've seen this normalized title before
        existing_post = posts_by_norm_title.get(norm_title)

        if existing_post is None:
            # First time seeing this title, add it
            posts_by_norm_title[norm_title] = p
        else:
            # Duplicate title found, compare scores
            existing_score = safe_int_or_zero(existing_post.get("baseScore"))
            if current_score > existing_score:
                # Current post has a higher score, replace the existing one
                posts_by_norm_title[norm_title] = p
            # Otherwise, keep the existing higher-scoring post

    unique_posts = list(posts_by_norm_title.values())
    duplicates_removed = valid_post_count - len(unique_posts)

    if discarded_no_title > 0:
        print(f"--- Discarded {discarded_no_title} posts lacking a title during deduplication ---")
    print(f"--- Kept {len(unique_posts)} unique posts (removed {duplicates_removed} lower-scoring duplicates) ---")
    logging.info(f"Deduplication: Input {len(posts)}, Valid w/ Title {valid_post_count}, Unique Output {len(unique_posts)}")
    return unique_posts

# ================================================================
#                     Main Processing Logic
# ================================================================

def main(limit: int | None = None):
    """
    Main execution function (limit: process at most N new posts; for canary runs).
    1. Fetches posts from EA, LW, AF.
    2. Filters posts based on criteria.
    3. Deduplicates posts by title, keeping highest score.
    4. Connects to the database.
    5. Fetches existing titles to avoid reprocessing.
    6. Processes each unique post: cleans HTML, converts to Markdown, runs Azure OpenAI analyses.
    7. Inserts processed data into the database in batches.
    8. Prints summary statistics.
    """
    start_time = time.time()
    print("================================================================")
    print(f"Starting AI Safety Feed Ingestion Script at {datetime.now(timezone.utc)}")
    print("================================================================")

    # -------- 1. Fetch Raw Data --------
    print("\n--- Fetching Raw Posts ---")
    ea_raw = get_forum_posts(EA_API_URL, EA_AI_SAFETY_TAG_ID, limit=DEFAULT_LIMIT)
    lw_raw = get_forum_posts(LW_API_URL, LW_AI_SAFETY_TAG_ID, limit=DEFAULT_LIMIT)
    # Alignment Forum posts arrive via the LessWrong query with af=true (see filter_lw_posts).
    # The separate AF endpoint was rate-limited on every run and is no longer queried.

    # -------- 2. Filter Data --------
    print("\n--- Filtering Posts ---")
    ea_posts = filter_ea_posts(ea_raw, EA_AI_SAFETY_TAG_ID)
    lw_posts = filter_lw_posts(lw_raw)

    # -------- 3. Combine & Deduplicate --------
    combined_filtered_posts = ea_posts + lw_posts
    initial_filtered_count = len(combined_filtered_posts)
    print(f"\n--- Total posts from all sources after initial filtering: {initial_filtered_count} ---")

    unique_posts = choose_highest_score(combined_filtered_posts)
    total_unique_count = len(unique_posts)

    if not unique_posts:
        print("\nNo unique posts remaining after filtering and deduplication. Nothing to process. Exiting.")
        logging.info("No unique posts remaining after filtering and deduplication. Exiting.")
        return # Exit gracefully if no posts left

    # -------- 4. Database Connection & Setup --------
    conn = None
    processed_count = 0
    affected_rows_count = 0 # Tracks total rows successfully inserted/updated
    failed_analysis_count = 0 # Track posts where *any* analysis step failed/skipped
    batch_data = [] # List to hold data tuples for batch insert
    total_db_failures = 0 # Track rows in failed batches
    embedding_failures_count = 0 # Track embedding generation failures
    total_skipped_recorded_in_db_count = 0 # New counter for skips recorded in DB
    new_posts_started = 0 # For --limit

    try:
        print("\n--- Connecting to Database ---")
        logging.info(f"Connecting to database using URL: {DATABASE_URL[:20]}...") # Log partial URL
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False # Ensure transactions are handled manually
        register_vector(conn) # <<< REGISTER VECTOR TYPE HANDLER
        print("Database connection successful.")
        logging.info("Database connection successful.")

        # -------- 5. Fetch Existing Titles --------
        with conn.cursor() as cur: # Use 'with' for automatic cursor closing
            print("Fetching existing normalized titles from database...")
            cur.execute("SELECT title_norm FROM content")
            # Fetchall can be memory intensive for huge tables, but likely okay here.
            # Consider server-side cursors or LIMIT/OFFSET for very large tables.
            existing_titles = {row[0] for row in cur.fetchall()}
            print(f"--> Found {len(existing_titles):,} existing titles in the database.")
            logging.info(f"Fetched {len(existing_titles)} existing titles.")

            # Fetch already skipped titles
            print("Fetching already skipped normalized titles from database...")
            # Cache BOTH keys: skipped_posts' primary key is post_id, so a retitled post must
            # still be recognised (otherwise the skip insert collides on the PK).
            cur.execute("SELECT post_id, title_norm FROM skipped_posts")
            skipped_rows = cur.fetchall()
            already_skipped = {r[1] for r in skipped_rows if r[1]}
            skipped_post_ids = {r[0] for r in skipped_rows if r[0]}
            print(f"--> Found {len(already_skipped):,} already skipped titles in the database.")
            logging.info(f"Fetched {len(already_skipped)} already skipped titles.")

            # -------- 6. Process Posts & Perform Analysis --------
            print(f"\n--- Starting Processing and Analysis for {total_unique_count} Unique Posts ---")
            for i, post in enumerate(unique_posts):
                processed_count += 1
                post_id = post.get('_id', 'N/A') # For logging
                title = post.get('title', 'Untitled')
                url = post.get('pageUrl', 'N/A')
                print(f"\n[{processed_count}/{total_unique_count}] Processing Post: '{title[:70]}...' (ID: {post_id})")
                logging.info(f"Processing post {processed_count}/{total_unique_count}: ID {post_id}, Title: {title[:70]}...")

                # --- 6a. Early Skip: Check if already in DB or marked as skipped ---
                norm_title = normalise_title(title)
                if norm_title in already_skipped or str(post_id) in skipped_post_ids: # Check this first
                    print(f"  -> Skipping (already marked as skipped or failed this run): Normalized title '{norm_title}'.")
                    logging.info(f"  Skipping post ID {post_id} (Title: {title[:70]}...) - already marked as skipped (in skipped_posts or this run).")
                    continue # Skip to the next post
                if norm_title in existing_titles:
                    print(f"  -> Skipping (already in content table): Normalized title '{norm_title}'.")
                    logging.info(f"  Skipping post ID {post_id} (Title: {title[:70]}...) - already in content table.")
                    continue # Skip to the next post

                if limit is not None:
                    if new_posts_started >= limit:
                        print(f"  -> --limit {limit} reached; stopping before this post.")
                        break
                    new_posts_started += 1

                # --- 6b. Initialize Analysis Variables ---
                sentence_summary = None
                paragraph_summary = None
                key_implication = None
                db_cluster = None   # Store the extracted cluster string for DB
                db_tags = None      # Store the extracted tags list for DB
                full_content_markdown = None # Initialize markdown content
                embedding_short_vector = None # Initialize short embedding
                embedding_full_vector = None  # Initialize full embedding

                # --- 6c. Extract & Clean Content ---
                print("  -> Cleaning HTML and converting to Markdown...")
                html_body = post.get('htmlBody', '') or "" # Ensure it's a string

                # Prepend title for context (optional, but can help analysis)
                if title != 'Untitled' and html_body:
                    html_body = f"<h1>{title}</h1>\n\n{html_body}"

                cleaned_html = ""
                if html_body:
                    try:
                        soup = BeautifulSoup(html_body, 'html.parser')
                        # Remove script/style tags which interfere with markdown/analysis
                        for element in soup(["script", "style", "noscript"]):
                            element.decompose()
                        # Optional: Add more cleaning steps here (e.g., remove ads, specific divs)
                        cleaned_html = str(soup)
                        print("  -> HTML cleaning successful.")
                    except Exception as e:
                        logging.error(f"BeautifulSoup cleaning failed for post ID {post_id} ('{title[:50]}...'). Error: {e}", exc_info=True)
                        print(f"  ERROR: HTML cleaning failed: {e}. Marking for skip and continuing.")
                        record_skip(cur, post_id, norm_title, url)
                        conn.commit() # Commit the skip record
                        already_skipped.add(norm_title); skipped_post_ids.add(str(post_id))
                        total_skipped_recorded_in_db_count += 1
                        continue # Skip this post
                else:
                    print("  -> Post has no HTML body content. Marking for skip and continuing.")
                    logging.warning(f"Post ID {post_id} ('{title[:50]}...') has no HTML body. Marking for skip.")
                    record_skip(cur, post_id, norm_title, url)
                    conn.commit() # Commit the skip record
                    already_skipped.add(norm_title); skipped_post_ids.add(str(post_id))
                    total_skipped_recorded_in_db_count += 1
                    continue # Skip this post

                # Convert cleaned HTML to Markdown (only if cleaning succeeded and body existed)
                # If we reach here, HTML processing was okay.
                try:
                    full_content_markdown = markdownify(cleaned_html, heading_style="ATX", bullets="-")
                    print("  -> Markdown conversion successful.")
                except Exception as e:
                    logging.error(f"Markdownify conversion failed for post ID {post_id} ('{title[:50]}...'). Error: {e}", exc_info=True)
                    print(f"  ERROR: Markdown conversion failed: {e}. Marking for skip and continuing.")
                    record_skip(cur, post_id, norm_title, url)
                    conn.commit() # Commit the skip record
                    already_skipped.add(norm_title); skipped_post_ids.add(str(post_id))
                    total_skipped_recorded_in_db_count += 1
                    continue # Skip this post

                # --- 6d. LLM analysis: ONE structured call (see llm_common.analyze_content) ---
                # Any failure means the post is NOT inserted: a transient error leaves the
                # title out of both tables so the next run retries it; a content-filter
                # rejection is recorded in skipped_posts so we stop retrying it.
                print("  -> Performing Azure OpenAI analysis (single structured call)...")
                original_tags = [t.get("name", "N/A") for t in post.get("tags", []) if t]
                try:
                    analysis = analyze_content(title, full_content_markdown, original_tags)
                except LLMContentFiltered as e:
                    print(f"  -> Content filter blocked this post: {e}. Recording skip.")
                    logging.warning(f"Content filter blocked post ID {post_id} ('{title[:50]}...'): {e}")
                    record_skip(cur, post_id, norm_title, url)
                    conn.commit()
                    already_skipped.add(norm_title); skipped_post_ids.add(str(post_id))
                    total_skipped_recorded_in_db_count += 1
                    failed_analysis_count += 1
                    continue
                except (LLMError, ValueError) as e:
                    print(f"  -> Analysis failed: {e}. Not inserted; will retry next run.")
                    logging.error(f"Analysis failed for post ID {post_id} ('{title[:50]}...'): {e}")
                    failed_analysis_count += 1
                    continue

                sentence_summary = analysis["sentence_summary"]
                paragraph_summary = analysis["paragraph_summary"]
                key_implication = analysis["key_implication"]
                db_cluster = analysis["cluster_tag"]
                db_tags = analysis["tags"]
                print(f"    - Analysis complete: Cluster='{db_cluster}', Tags={db_tags}")

                # --- 6f. Extract Other Metadata ---
                print("  -> Extracting remaining metadata...")
                source_type = post.get('source_type', 'Unknown')
                score = safe_int_or_zero(post.get('baseScore'))
                comment_count = safe_int_or_zero(post.get('commentCount'))
                cleaned_title = None  # left NULL on insert; rewrite_titles.py --mode titles fills it (frontend/backend fall back to title)

                # Extract image URL (simple regex for first src or data-src)
                image_url = None
                if cleaned_html: # Use cleaned HTML to avoid script/style interference
                    match = re.search(r'<img[^>]+(?:src|data-src)=["\']([^"\']+)["\']', cleaned_html, re.IGNORECASE)
                    if match:
                        image_url = match.group(1)
                        print(f"    - Found image URL: {image_url[:60]}...")

                # Extract authors (using set for uniqueness)
                authors_set = set()
                if post.get('user') and post['user'].get('displayName'):
                    authors_set.add(post['user']['displayName'])
                if post.get('coauthors'):
                    for author in post['coauthors']:
                        if author and author.get('displayName'):
                            authors_set.add(author['displayName'])
                authors_set.discard(None) # Remove potential None values
                # Convert set to sorted list for consistent DB insertion (ARRAY type)
                authors_list = sorted(list(authors_set)) if authors_set else ['Unknown']
                print(f"    - Authors: {authors_list}")

                # Extract and parse publication date
                published_date = iso_to_dt(post.get('postedAt'))
                print(f"    - Published Date: {published_date}")

                # --- 6e. Generate Embeddings (title + analysis results) ---
                print("  -> Generating OpenAI embeddings...")
                embedding_short_vector, embedding_full_vector = generate_embeddings(
                    title or "",
                    build_embedding_text(sentence_summary, paragraph_summary, key_implication, db_tags),
                )
                if embedding_short_vector is None or embedding_full_vector is None:
                    print(f"    - Embedding generation failed for post ID {post_id}. Not inserted; will retry next run.")
                    logging.error(f"Embedding generation failed for post ID {post_id}")
                    embedding_failures_count += 1
                    continue
                print("    - Embeddings generated successfully.")

                # --- 6g. Prepare Data Tuple for Insertion ---
                # Ensure the order matches DB_COLS exactly!
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
                    html_body,                      # full_content (original HTML, maybe with prepended title)
                    full_content_markdown,          # full_content_markdown (str or None)
                    comment_count,                  # comment_count (int)
                    db_cluster,                     # cluster_tag (AI generated cluster or None)
                    embedding_short_vector,         # embedding_short (list[float] or None)
                    embedding_full_vector,          # embedding_full (list[float] or None)
                    cleaned_title,                  # cleaned_title (str or None)
                )

                if len(data_tuple) != NUM_DB_COLS:
                    logging.critical(f"Tuple length mismatch for post ID {post_id}: {len(data_tuple)} != {NUM_DB_COLS}. Skipping.")
                    continue

                # --- 6h. Add to Batch ---
                batch_data.append(data_tuple)
                print(f"  -> Added post '{title[:50]}...' to batch (Batch size: {len(batch_data)}).")

                # -------- 7. Insert Batch into Database --------
                if len(batch_data) >= BATCH_SIZE:
                    print(f"\n--- Executing database batch insert ({len(batch_data)} posts) ---")
                    batch_insert_successful = False
                    try:
                        # Use extras.execute_batch for efficient insertion
                        # The INSERT_SQL already handles ON CONFLICT DO NOTHING
                        cur.executemany(INSERT_SQL, batch_data)
                        conn.commit() # Commit the transaction for this batch
                        batch_insert_successful = True
                        print(f"--- Batch insert successful. {len(batch_data)} rows processed (inserted or skipped on conflict). ---")
                        logging.info(f"Successfully executed batch insert for {len(batch_data)} posts.")
                    except psycopg2.DatabaseError as db_err:
                        logging.error(f"Database error during batch execution: {db_err}", exc_info=True)
                        print(f"ERROR: Database error during batch execution: {db_err}. Rolling back batch.")
                        conn.rollback() # Rollback the failed batch transaction
                        total_db_failures += len(batch_data) # Increment failure count
                    except Exception as e:
                        logging.error(f"Unexpected error during batch execution: {e}", exc_info=True)
                        print(f"ERROR: Unexpected error during batch execution: {e}. Rolling back batch.")
                        conn.rollback() # Rollback on unexpected errors
                        total_db_failures += len(batch_data) # Increment failure count
                    finally:
                        # Update total affected rows only if batch succeeded
                        if batch_insert_successful:
                            affected_rows_count += len(batch_data)
                        # Always clear the batch list
                        batch_data = []

            # -------- 8. Insert Final Batch --------
            if batch_data:
                print(f"\n--- Executing final database batch insert ({len(batch_data)} posts) ---", flush=True)
                final_batch_successful = False
                try:
                    cur.executemany(INSERT_SQL, batch_data)
                    conn.commit() # Commit the final batch
                    final_batch_successful = True
                    print(f"--- Final batch insert successful. {len(batch_data)} rows processed. ---")
                    logging.info(f"Successfully executed final batch insert for {len(batch_data)} posts.")
                except psycopg2.DatabaseError as db_err:
                    logging.error(f"Database error during final batch execution: {db_err}", exc_info=True)
                    print(f"ERROR: Database error during final batch execution: {db_err}. Rolling back batch.")
                    conn.rollback()
                    total_db_failures += len(batch_data)
                except Exception as e:
                    logging.error(f"Unexpected error during final batch execution: {e}", exc_info=True)
                    print(f"ERROR: Unexpected error during final batch execution: {e}. Rolling back batch.")
                    conn.rollback()
                    total_db_failures += len(batch_data)
                finally:
                    if final_batch_successful:
                        affected_rows_count += len(batch_data)
                    batch_data = [] # Clear list

    except psycopg2.OperationalError as e:
        # Errors during connection itself
        logging.critical(f"FATAL: Database connection failed: {e}", exc_info=True)
        print(f"FATAL ERROR: Database connection failed: {e}")
        # Cannot proceed without DB connection
    except psycopg2.DatabaseError as e:
        # Other database errors (e.g., during initial title fetch)
        logging.error(f"Database error occurred outside batch processing: {e}", exc_info=True)
        print(f"ERROR: Database error occurred: {e}")
        if conn:
            conn.rollback() # Rollback any potential changes if connection exists
    except Exception as e:
        # Catch-all for unexpected errors in the main processing block
        logging.error(f"An unexpected error occurred in the main processing loop: {e}", exc_info=True)
        print(f"ERROR: An unexpected error occurred: {e}")
        if conn:
            conn.rollback() # Rollback on general errors too
    finally:
        # -------- 9. Close Database Connection --------
        if conn:
            conn.close()
            print("\nDatabase connection closed.")
            logging.info("Database connection closed.")

    # -------- 10. Print Final Summary --------
    end_time = time.time()
    duration = end_time - start_time
    print("\n================================================================")
    print("--- Processing Summary ---")
    print("================================================================")
    print(f"Total posts initially combined from filtered sources: {initial_filtered_count}")
    print(f"Unique posts after deduplication: {total_unique_count}")
    print(f"Posts processed (attempted analysis/DB insert): {processed_count}")
    print(f"Posts with failed/skipped Azure OpenAI analysis step(s): {failed_analysis_count}")
    print(f"Posts with failed/skipped OpenAI embedding generation: {embedding_failures_count}")
    print(f"Posts recorded in 'skipped_posts' table this run: {total_skipped_recorded_in_db_count}") # New summary line
    print(f"Total rows processed in successful DB batches: {affected_rows_count}")
    print(f"Estimated rows in failed DB batches: {total_db_failures}")
    print(f"Script finished at {datetime.now(timezone.utc)}")
    print(f"Total execution time: {duration:.2f} seconds")
    print("================================================================")
    logging.info(f"Script finished. Duration: {duration:.2f}s. Processed: {processed_count}. DB Success: {affected_rows_count}. DB Fail: {total_db_failures}. Analysis Fail: {failed_analysis_count}. Embedding Fail: {embedding_failures_count}. Recorded Skips: {total_skipped_recorded_in_db_count}.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="EA Forum / LessWrong / Alignment Forum ingestion")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N new posts (canary runs)")
    main(limit=ap.parse_args().limit)
