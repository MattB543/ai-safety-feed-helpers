#!/usr/bin/env python3
"""
classify_twitter_ai_safety.py
─────────────────────────────
Classify Twitter conversations as AI Safety related using Gemini (gemini-3.8-flash).

This script:
1. Converts Twitter conversations to clean markdown
2. Uses Gemini Flash to classify if AI safety related
3. Stores is_ai_safety boolean per conversation_id

Usage:
    python classify_twitter_ai_safety.py --test-markdown
    python classify_twitter_ai_safety.py --test-classify 50
    python classify_twitter_ai_safety.py --estimate --month 2025-11

Requirements:
    pip install psycopg2-binary python-dotenv google-genai
"""

import argparse
import os
import sys
import logging
import time
import threading
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Fix Windows console encoding issues with Unicode characters
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

load_dotenv(override=True)

DATABASE_URL = os.getenv("AI_SAFETY_FEED_DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_TEXT_MODEL") or "gemini-3.8-flash"

if not DATABASE_URL:
    print("ERROR: AI_SAFETY_FEED_DB_URL environment variable not set.")
    sys.exit(1)

# ───── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.ERROR)

# ───── Rate Limiter ───────────────────────────────────────────────────────────

class RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, rpm: int = 10):
        """
        Initialize rate limiter.

        Args:
            rpm: Requests per minute limit (default 10 for free tier)
        """
        self.rpm = rpm
        self.interval = 60.0 / rpm  # Minimum seconds between requests
        self.lock = threading.Lock()
        self.last_request = 0.0

    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                time.sleep(sleep_time)
            self.last_request = time.time()


# Global rate limiter (will be initialized based on --rpm flag)
rate_limiter = None


# ───── Gemini client (lazy init) ──────────────────────────────────────────────
gemini_client = None

def get_gemini_client():
    """Lazily initialize and return the Gemini client."""
    global gemini_client
    if gemini_client is None:
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set - classification will be skipped")
            return None
        try:
            from google import genai
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Gemini client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return None
    return gemini_client


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_classified_conversation_ids(conn) -> set:
    """Get set of conversation_ids that have already been classified."""
    with conn.cursor() as cur:
        cur.execute("SELECT conversation_id FROM twitter_conversation_classifications")
        return {row[0] for row in cur.fetchall()}


def save_classification(conn, conversation_id: str, is_ai_safety: bool, model: str = GEMINI_MODEL):
    """Save a classification result to the database."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO twitter_conversation_classifications (conversation_id, is_ai_safety, model_used)
            VALUES (%s, %s, %s)
            ON CONFLICT (conversation_id) DO UPDATE SET
                is_ai_safety = EXCLUDED.is_ai_safety,
                classified_at = NOW(),
                model_used = EXCLUDED.model_used
        """, (conversation_id, is_ai_safety, model))
    conn.commit()


def get_conversation_ids(conn, month: Optional[str] = None, limit: Optional[int] = None) -> list:
    """
    Get unique conversation_ids from the database.

    Args:
        conn: Database connection
        month: Optional month filter in format 'YYYY-MM'
        limit: Optional limit on number of conversations

    Returns:
        List of conversation_id strings
    """
    with conn.cursor() as cur:
        query = """
            SELECT conversation_id, COUNT(*) as tweet_count
            FROM tweets
            WHERE conversation_id IS NOT NULL
        """
        params = []

        if month:
            year, mon = month.split('-')
            start_date = f"{year}-{mon}-01"
            # Calculate next month
            next_mon = int(mon) + 1
            next_year = int(year)
            if next_mon > 12:
                next_mon = 1
                next_year += 1
            end_date = f"{next_year}-{next_mon:02d}-01"
            query += " AND created_at >= %s AND created_at < %s"
            params.extend([start_date, end_date])

        query += " GROUP BY conversation_id ORDER BY MAX(created_at) DESC"

        if limit:
            query += f" LIMIT {limit}"

        cur.execute(query, params)
        return [(row[0], row[1]) for row in cur.fetchall()]


def get_conversation_tweets(conn, conversation_id: str) -> list:
    """
    Get all tweets for a conversation, ordered chronologically.

    Returns list of dicts with tweet data.
    """
    with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                tweet_id,
                author_username,
                author_display_name,
                text,
                created_at,
                interaction_type,
                in_reply_to_tweet_id,
                in_reply_to_username,
                quoted_text,
                quoted_username,
                retweeted_text,
                retweeted_username
            FROM tweets
            WHERE conversation_id = %s
            ORDER BY created_at ASC
        """, (conversation_id,))
        return [dict(row) for row in cur.fetchall()]


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def conversation_to_markdown(tweets: list, max_chars: int = 100000) -> str:
    """
    Convert a list of tweets (one conversation) to clean, minimal markdown
    with tree-style visual hierarchy.

    Args:
        tweets: List of tweet dicts from get_conversation_tweets()
        max_chars: Maximum characters for the output (for truncation)

    Returns:
        Markdown string representation of the conversation
    """
    if not tweets:
        return ""

    # Build lookup structures
    tweets_by_id = {t['tweet_id']: t for t in tweets}
    children_by_parent = {}  # parent_tweet_id -> list of child tweets

    # Find root(s) and build parent-child relationships
    roots = []
    for tweet in tweets:
        reply_to_id = tweet.get('in_reply_to_tweet_id')
        if reply_to_id and reply_to_id in tweets_by_id:
            # This tweet replies to another tweet in our set
            if reply_to_id not in children_by_parent:
                children_by_parent[reply_to_id] = []
            children_by_parent[reply_to_id].append(tweet)
        else:
            # This is a root (no parent in our set)
            roots.append(tweet)

    # Sort roots by created_at
    roots.sort(key=lambda t: t.get('created_at') or datetime.min.replace(tzinfo=timezone.utc))

    # Sort children by created_at
    for parent_id in children_by_parent:
        children_by_parent[parent_id].sort(
            key=lambda t: t.get('created_at') or datetime.min.replace(tzinfo=timezone.utc)
        )

    def format_tweet_text(tweet: dict) -> str:
        """Format a single tweet's content."""
        author = tweet.get('author_username') or 'unknown'
        text = tweet.get('text') or ''
        interaction = tweet.get('interaction_type') or 'tweet'

        if interaction == 'retweet':
            rt_user = tweet.get('retweeted_username') or 'unknown'
            rt_text = tweet.get('retweeted_text') or ''
            return f"@{author} RT @{rt_user}: {rt_text}"

        elif interaction == 'quote_tweet':
            qt_user = tweet.get('quoted_username') or 'unknown'
            qt_text = tweet.get('quoted_text') or ''
            if text:
                return f"@{author}: {text} [quoting @{qt_user}: {qt_text}]"
            else:
                return f"@{author} quoting @{qt_user}: {qt_text}"

        else:  # Regular tweet or reply
            return f"@{author}: {text}"

    def render_tree(tweet: dict, prefix: str = "", is_last: bool = True, is_root: bool = False) -> list:
        """Recursively render a tweet and its replies as a tree."""
        lines = []

        # Format the tweet content (handle multi-line by joining)
        content = format_tweet_text(tweet)
        # Replace newlines with spaces for cleaner tree display
        content = ' '.join(content.split())

        # Add this tweet
        if is_root:
            # Root tweet - no tree chars
            lines.append(content)
        else:
            # Reply - use tree chars
            connector = "└─ " if is_last else "├─ "
            lines.append(prefix + connector + content)

        # Process children
        tweet_id = tweet.get('tweet_id')
        children = children_by_parent.get(tweet_id, [])

        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            if is_root:
                # Children of root - no prefix yet
                child_prefix = ""
            else:
                # Deeper nesting - extend prefix
                child_prefix = prefix + ("   " if is_last else "│  ")

            child_lines = render_tree(child, child_prefix, child_is_last, is_root=False)
            lines.extend(child_lines)

        return lines

    # Render all root tweets and their trees
    all_lines = []
    for i, root in enumerate(roots):
        tree_lines = render_tree(root, "", True, is_root=True)
        all_lines.extend(tree_lines)
        if i < len(roots) - 1:
            all_lines.append("")  # Blank line between separate threads

    # Join and truncate if needed
    result = '\n'.join(all_lines)

    if len(result) > max_chars:
        result = result.encode('utf-8')[:max_chars].decode('utf-8', 'ignore')
        result += "\n\n[... truncated]"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# AI SAFETY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def is_ai_safety_conversation(markdown: str, model: str = GEMINI_MODEL) -> Optional[bool]:
    """
    Classify if a conversation is about AI safety using Gemini Flash.

    Args:
        markdown: Markdown representation of the conversation
        model: Gemini model to use

    Returns:
        True if AI safety related, False if not, None on error
    """
    client = get_gemini_client()
    if not client:
        logger.warning("Gemini client not available - skipping classification")
        return None

    # Truncate to 100k chars for the prompt (Gemini handles ~1M tokens)
    snippet = markdown.encode('utf-8')[:100000].decode('utf-8', 'ignore')

    prompt = f"""You are an expert AI-safety content curator.
Answer YES or NO – nothing else.

Does this Twitter conversation primarily discuss AI safety or closely related topics
(alignment, risk, governance, technical ML safety, policy, x-risk, interpretability, etc.)?

If ANY tweet in the thread is substantively about AI safety topics, answer YES.

Conversation:
{snippet}
"""

    try:
        from google.genai import types

        # Apply rate limiting if enabled
        global rate_limiter
        if rate_limiter:
            rate_limiter.wait()

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            )
        )
        answer = (response.text or "").strip().upper()
        # Exact-prefix match: "NO, although ... YES" must not count as YES.
        return answer.startswith("YES")

    except Exception as e:
        logger.warning(f"Classification failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND MODES
# ═══════════════════════════════════════════════════════════════════════════════

def test_markdown_mode(conn):
    """
    Test markdown conversion by saving sample conversations to files.
    Saves varied sizes: small (1-2 tweets), medium (3-5), large (6+).
    """
    output_dir = Path("conversation_samples")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("TEST MARKDOWN MODE")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}\n")

    with conn.cursor() as cur:
        # Get sample conversations of different sizes

        # Small conversations (1-2 tweets)
        cur.execute("""
            SELECT conversation_id, COUNT(*) as cnt
            FROM tweets
            WHERE conversation_id IS NOT NULL
              AND created_at >= '2025-11-01' AND created_at < '2025-12-01'
            GROUP BY conversation_id
            HAVING COUNT(*) BETWEEN 1 AND 2
            ORDER BY RANDOM()
            LIMIT 5
        """)
        small = cur.fetchall()

        # Medium conversations (3-5 tweets)
        cur.execute("""
            SELECT conversation_id, COUNT(*) as cnt
            FROM tweets
            WHERE conversation_id IS NOT NULL
              AND created_at >= '2025-11-01' AND created_at < '2025-12-01'
            GROUP BY conversation_id
            HAVING COUNT(*) BETWEEN 3 AND 5
            ORDER BY RANDOM()
            LIMIT 5
        """)
        medium = cur.fetchall()

        # Large conversations (6+ tweets)
        cur.execute("""
            SELECT conversation_id, COUNT(*) as cnt
            FROM tweets
            WHERE conversation_id IS NOT NULL
              AND created_at >= '2025-11-01' AND created_at < '2025-12-01'
            GROUP BY conversation_id
            HAVING COUNT(*) >= 6
            ORDER BY RANDOM()
            LIMIT 5
        """)
        large = cur.fetchall()

    all_samples = [
        ("small", small),
        ("medium", medium),
        ("large", large)
    ]

    saved_count = 0
    for category, samples in all_samples:
        print(f"\n--- {category.upper()} conversations ({len(samples)} samples) ---")

        for conv_id, tweet_count in samples:
            tweets = get_conversation_tweets(conn, conv_id)
            markdown = conversation_to_markdown(tweets)

            # Save to file
            filename = output_dir / f"{category}_{conv_id}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Conversation {conv_id}\n")
                f.write(f"**Tweet count:** {tweet_count}\n")
                f.write(f"**Characters:** {len(markdown)}\n\n")
                f.write("---\n\n")
                f.write(markdown)

            print(f"  Saved: {filename.name} ({tweet_count} tweets, {len(markdown)} chars)")
            saved_count += 1

    print(f"\n{'=' * 60}")
    print(f"Saved {saved_count} sample conversations to {output_dir}/")
    print("Please review these files to validate the markdown format.")
    print("=" * 60)


def test_classify_mode(conn, count: int = 50):
    """
    Test classification on N conversations, showing results for validation.
    """
    print("\n" + "=" * 60)
    print(f"TEST CLASSIFY MODE ({count} conversations)")
    print("=" * 60)

    # Get random November conversations
    conversations = get_conversation_ids(conn, month="2025-11", limit=count)

    if not conversations:
        print("No conversations found for the specified criteria.")
        return

    print(f"Processing {len(conversations)} conversations...\n")

    results = {"yes": 0, "no": 0, "error": 0}

    for i, (conv_id, tweet_count) in enumerate(conversations):
        tweets = get_conversation_tweets(conn, conv_id)
        markdown = conversation_to_markdown(tweets)

        # Classify
        is_ai_safety = is_ai_safety_conversation(markdown)

        # Track results
        if is_ai_safety is True:
            results["yes"] += 1
            label = "YES"
        elif is_ai_safety is False:
            results["no"] += 1
            label = "NO"
        else:
            results["error"] += 1
            label = "ERR"

        # Show snippet
        first_tweet_text = tweets[0].get('text', '')[:80] if tweets else ''
        author = tweets[0].get('author_username', 'unknown') if tweets else 'unknown'

        print(f"[{i+1:3d}/{len(conversations)}] {label:3s} | @{author}: {first_tweet_text}...")

    print(f"\n{'=' * 60}")
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"  AI Safety (YES): {results['yes']}")
    print(f"  Not AI Safety (NO): {results['no']}")
    print(f"  Errors: {results['error']}")
    print(f"  Total: {sum(results.values())}")
    print("=" * 60)


def get_engaged_conversation_ids(conn, month: str) -> list:
    """
    Get conversation_ids that have at least one tweet with significant engagement:
    - 20+ likes OR 5+ retweets OR 5+ quote tweets OR 10+ replies

    Returns list of (conversation_id, tweet_count) tuples.
    """
    year, mon = month.split('-')
    start_date = f"{year}-{mon}-01"
    next_mon = int(mon) + 1
    next_year = int(year)
    if next_mon > 12:
        next_mon = 1
        next_year += 1
    end_date = f"{next_year}-{next_mon:02d}-01"

    with conn.cursor() as cur:
        cur.execute("""
            SELECT conversation_id, COUNT(*) as tweet_count
            FROM tweets
            WHERE conversation_id IS NOT NULL
              AND created_at >= %s AND created_at < %s
              AND conversation_id IN (
                  SELECT DISTINCT conversation_id
                  FROM tweets
                  WHERE created_at >= %s AND created_at < %s
                    AND (favorite_count >= 20
                         OR retweet_count >= 5
                         OR quote_count >= 5
                         OR reply_count >= 10)
              )
            GROUP BY conversation_id
            ORDER BY MAX(created_at) DESC
        """, (start_date, end_date, start_date, end_date))
        return [(row[0], row[1]) for row in cur.fetchall()]


def estimate_mode(conn, month: str = "2025-11"):
    """
    Estimate token cost for processing engaged conversations in a month.
    Only includes conversations with at least one tweet having:
    - 10+ likes OR 3+ retweets OR 3+ quote tweets OR 5+ replies
    """
    print("\n" + "=" * 60)
    print(f"ESTIMATE MODE (month: {month})")
    print("=" * 60)
    print("Filter: 20+ likes OR 5+ RTs OR 5+ quotes OR 10+ replies")
    print("=" * 60)

    # Get all conversations (for comparison)
    all_conversations = get_conversation_ids(conn, month=month)
    total_all = len(all_conversations)

    # Get engaged conversations only
    conversations = get_engaged_conversation_ids(conn, month=month)

    if not conversations:
        print("No engaged conversations found for the specified month.")
        return

    total_conversations = len(conversations)
    print(f"All conversations in {month}: {total_all:,}")
    print(f"Engaged conversations: {total_conversations:,} ({100*total_conversations/total_all:.1f}%)")

    # Sample up to 200 conversations to estimate average size
    sample_size = min(200, total_conversations)
    sample = conversations[:sample_size]

    total_chars = 0
    total_tweets = 0

    print(f"Sampling {sample_size} conversations to estimate size...")

    for conv_id, tweet_count in sample:
        tweets = get_conversation_tweets(conn, conv_id)
        markdown = conversation_to_markdown(tweets)
        total_chars += len(markdown)
        total_tweets += tweet_count

    avg_chars_per_conv = total_chars / sample_size
    avg_tweets_per_conv = total_tweets / sample_size

    # Extrapolate to full set
    estimated_total_chars = avg_chars_per_conv * total_conversations
    estimated_total_tokens = estimated_total_chars / 4  # 4 chars per token estimate

    # Gemini Flash pricing (approximate)
    # Input: $0.075 per 1M tokens
    cost_per_million = 0.075
    estimated_cost = (estimated_total_tokens / 1_000_000) * cost_per_million

    print("\n" + "=" * 60)
    print("ESTIMATION RESULTS")
    print("=" * 60)
    print(f"  Engaged conversations: {total_conversations:,}")
    print(f"  Avg tweets/conversation: {avg_tweets_per_conv:.1f}")
    print(f"  Avg chars/conversation: {avg_chars_per_conv:.0f}")
    print(f"  ")
    print(f"  Estimated total chars: {estimated_total_chars:,.0f}")
    print(f"  Estimated total tokens: {estimated_total_tokens:,.0f}")
    print(f"  Estimated cost (Gemini Flash): ${estimated_cost:.3f}")
    print("=" * 60)


def export_ai_safety_mode(conn, month: Optional[str] = None, output_file: str = "ai_safety_conversations.md"):
    """
    Export all AI Safety classified conversations to a single markdown file,
    sorted chronologically.

    Args:
        conn: Database connection
        month: Optional month filter in YYYY-MM format
        output_file: Output file path
    """
    print("\n" + "=" * 60)
    print(f"EXPORT AI SAFETY CONVERSATIONS")
    print("=" * 60)

    # Build query to get AI Safety conversations with their earliest tweet date
    query = """
        SELECT
            c.conversation_id,
            MIN(t.created_at) as first_tweet_at,
            COUNT(t.tweet_id) as tweet_count
        FROM twitter_conversation_classifications c
        JOIN tweets t ON c.conversation_id = t.conversation_id
        WHERE c.is_ai_safety = TRUE
    """
    params = []

    if month:
        year, mon = month.split('-')
        start_date = f"{year}-{mon}-01"
        next_mon = int(mon) + 1
        next_year = int(year)
        if next_mon > 12:
            next_mon = 1
            next_year += 1
        end_date = f"{next_year}-{next_mon:02d}-01"
        query += " AND t.created_at >= %s AND t.created_at < %s"
        params.extend([start_date, end_date])
        print(f"Month filter: {month}")

    query += " GROUP BY c.conversation_id ORDER BY MIN(t.created_at) ASC"

    with conn.cursor() as cur:
        cur.execute(query, params)
        conversations = cur.fetchall()

    total = len(conversations)
    print(f"Found {total:,} AI Safety conversations to export")

    if not conversations:
        print("No AI Safety conversations found.")
        return

    # Build the markdown file
    print(f"\nGenerating markdown...")

    lines = []
    lines.append("# AI Safety Twitter Conversations")
    lines.append("")
    if month:
        lines.append(f"**Month:** {month}")
    lines.append(f"**Total conversations:** {total:,}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, (conv_id, first_tweet_at, tweet_count) in enumerate(conversations):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1:,}/{total:,}...")

        # Get tweets for this conversation
        tweets = get_conversation_tweets(conn, conv_id)

        if not tweets:
            continue

        # Get metadata
        first_author = tweets[0].get('author_username', 'unknown')
        date_str = first_tweet_at.strftime('%Y-%m-%d %H:%M') if first_tweet_at else 'unknown'

        # Generate markdown for conversation
        markdown = conversation_to_markdown(tweets)

        # Add to output with header
        lines.append(f"## {i + 1}. @{first_author} ({date_str})")
        lines.append(f"*Conversation ID: {conv_id} | {tweet_count} tweets*")
        lines.append("")
        lines.append("```")
        lines.append(markdown)
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"EXPORT COMPLETE")
    print("=" * 60)
    print(f"  Output file: {output_path.absolute()}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Conversations exported: {total:,}")
    print("=" * 60)


def run_mode(conn, month: str = "2025-11", limit: Optional[int] = None, workers: int = 1):
    """
    Run classification on engaged conversations and save results to DB.

    Args:
        conn: Database connection
        month: Month to process in YYYY-MM format
        limit: Optional limit on number of conversations to process
        workers: Number of parallel workers (default 1 for sequential)
    """
    print("\n" + "=" * 60)
    print(f"RUN MODE (month: {month})")
    print("=" * 60)
    print("Filter: 20+ likes OR 5+ RTs OR 5+ quotes OR 10+ replies")
    if limit:
        print(f"Limit: {limit} conversations")
    print(f"Workers: {workers} (parallel)" if workers > 1 else "Workers: 1 (sequential)")
    if rate_limiter:
        print(f"Rate limit: {rate_limiter.rpm} RPM")
    print("=" * 60)

    # Get engaged conversations
    conversations = get_engaged_conversation_ids(conn, month)

    if not conversations:
        print("No engaged conversations found for the specified month.")
        return

    print(f"Found {len(conversations):,} engaged conversations in {month}")

    # Get already classified
    already_classified = get_classified_conversation_ids(conn)
    print(f"Already classified: {len(already_classified):,}")

    # Filter out already classified
    to_process = [(cid, cnt) for cid, cnt in conversations if cid not in already_classified]
    print(f"Remaining to classify: {len(to_process):,}")

    if not to_process:
        print("\nAll conversations already classified!")
        return

    # Apply limit if specified
    if limit:
        to_process = to_process[:limit]
        print(f"Processing {len(to_process)} conversations (limited)")

    total_count = len(to_process)
    print(f"\nProcessing {total_count} conversations...\n")

    # Thread-safe counters and locks
    results = {"yes": 0, "no": 0, "error": 0}
    results_lock = threading.Lock()
    db_lock = threading.Lock()
    counter = {"done": 0}
    start_time = time.time()

    def process_conversation(conv_data):
        """Worker function to process a single conversation."""
        conv_id, tweet_count = conv_data

        # Get tweets (DB read - use lock)
        with db_lock:
            tweets = get_conversation_tweets(conn, conv_id)

        markdown = conversation_to_markdown(tweets)

        # Classify (API call - rate limiter handles timing)
        is_ai_safety = is_ai_safety_conversation(markdown)

        # Get author and snippet for logging
        author = tweets[0].get('author_username', 'unknown') if tweets else 'unknown'
        first_tweet_text = tweets[0].get('text', '')[:50] if tweets else ''

        # Update DB and counters (use locks)
        with results_lock:
            counter["done"] += 1
            current = counter["done"]

            if is_ai_safety is True:
                results["yes"] += 1
                label = "YES"
            elif is_ai_safety is False:
                results["no"] += 1
                label = "NO"
            else:
                results["error"] += 1
                label = "ERR"

        # Save to DB (if not error)
        if is_ai_safety is not None:
            with db_lock:
                save_classification(conn, conv_id, is_ai_safety)

        # Calculate rate
        elapsed = time.time() - start_time
        rate = current / elapsed * 60 if elapsed > 0 else 0

        print(f"[{current:4d}/{total_count}] {label:3s} | @{author}: {first_tweet_text}... ({rate:.1f}/min)")

        return (conv_id, is_ai_safety)

    # Run with ThreadPoolExecutor for parallel processing
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_conversation, conv) for conv in to_process]
            # Wait for all to complete
            concurrent.futures.wait(futures)
    else:
        # Sequential processing
        for conv in to_process:
            process_conversation(conv)

    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"  AI Safety (YES): {results['yes']}")
    print(f"  Not AI Safety (NO): {results['no']}")
    print(f"  Errors (not saved): {results['error']}")
    print(f"  Total processed: {sum(results.values())}")
    print(f"  Time elapsed: {elapsed_total:.1f}s ({sum(results.values())/elapsed_total*60:.1f}/min)")
    print("=" * 60)

    # Show DB totals
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM twitter_conversation_classifications WHERE is_ai_safety = TRUE")
        total_yes = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM twitter_conversation_classifications WHERE is_ai_safety = FALSE")
        total_no = cur.fetchone()[0]

    print(f"\nDatabase totals:")
    print(f"  AI Safety conversations: {total_yes}")
    print(f"  Not AI Safety: {total_no}")
    print(f"  Total classified: {total_yes + total_no}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Classify Twitter conversations as AI Safety related",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Test markdown conversion
  python classify_twitter_ai_safety.py --test-markdown

  # Step 2: Test classification on 50 conversations
  python classify_twitter_ai_safety.py --test-classify 50

  # Step 3: Estimate token cost for November
  python classify_twitter_ai_safety.py --estimate --month 2025-11

  # Step 4: Run classification and save to DB (with limit for testing)
  python classify_twitter_ai_safety.py --run --month 2025-11 --limit 50

  # Step 5: Run full classification for a month (sequential)
  python classify_twitter_ai_safety.py --run --month 2025-11

  # Step 6: Run with parallel workers (faster, requires paid API tier)
  python classify_twitter_ai_safety.py --run --month 2025-11 --workers 5 --rpm 100
"""
    )

    parser.add_argument(
        "--test-markdown",
        action="store_true",
        help="Test markdown conversion by saving sample conversations to files"
    )
    parser.add_argument(
        "--test-classify",
        type=int,
        metavar="N",
        help="Test classification on N random conversations"
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate token cost for processing all conversations in a month"
    )
    parser.add_argument(
        "--month",
        type=str,
        default="2025-11",
        help="Month to process in YYYY-MM format (default: 2025-11)"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run classification on engaged conversations and save to DB"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export all AI Safety classified conversations to a markdown file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ai_safety_conversations.md",
        metavar="FILE",
        help="Output file for --export mode (default: ai_safety_conversations.md)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Limit number of conversations to process (for --run mode)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers (default: 1). Use with --rpm for rate limiting."
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=10,
        metavar="N",
        help="API rate limit in requests per minute (default: 10 for free tier, use 100-1000 for paid)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Require at least one mode
    if not any([args.test_markdown, args.test_classify, args.estimate, args.run, args.export]):
        parser.print_help()
        print("\nError: Please specify a mode (--test-markdown, --test-classify, --estimate, --run, or --export)")
        sys.exit(1)

    # Initialize rate limiter
    global rate_limiter
    if args.rpm > 0:
        rate_limiter = RateLimiter(rpm=args.rpm)
        logger.info(f"Rate limiter initialized: {args.rpm} RPM")

    # Connect to database
    conn = None
    try:
        logger.info("Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        logger.info("Database connected")

        if args.test_markdown:
            test_markdown_mode(conn)

        if args.test_classify:
            test_classify_mode(conn, args.test_classify)

        if args.estimate:
            estimate_mode(conn, args.month)

        if args.run:
            run_mode(conn, args.month, args.limit, args.workers)

        if args.export:
            export_ai_safety_mode(conn, args.month, args.output)

    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()
