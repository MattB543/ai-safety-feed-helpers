#!/usr/bin/env python3
"""
Twitter Timeline Scraper for AI-Safety Feed (Fixed)
===================================================

Features:
- Scrapes Twitter Following timeline and thread content
- AI safety filtering using OpenAI GPT models
- Thread-based filtering: When processing threads, concatenates all tweets in a thread 
  and does ONE AI safety check per thread. If the thread passes, all tweets are inserted.
  If the thread fails, all tweets are added to the skipped_tweets table.
- Individual tweet filtering: For non-thread content, each tweet is filtered individually
- Re-checking system for previously skipped tweets that gain popularity
- Comprehensive logging and statistics tracking
"""
import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from dotenv import load_dotenv
from twitter.account import Account
from twitter.scraper import Scraper
import psycopg2
from psycopg2.extras import execute_values
from collections.abc import Generator

# OpenAI imports for AI safety filtering
from openai import OpenAI
from openai import APIError as OpenAI_APIError, RateLimitError as OpenAI_RateLimitError

# ---------------------------------------------------------------------------
# Timeline helper
# ---------------------------------------------------------------------------
def iter_timeline_entries(page: dict) -> Generator[dict, None, None]:
    """
    Yield every 'entry' object inside one GraphQL page,
    no matter whether it is in AddEntries, ReplaceEntry, or a Module.
    """
    stack = [page]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            # The two keys we care about
            if node.get("type") == "TimelineAddEntries" and "entries" in node:
                stack.extend(node["entries"])
            elif "entries" in node:                         # ReplaceEntry, etc.
                stack.extend(node["entries"])
            elif "items" in node:                           # TimelineModule
                stack.extend(node["items"])
            else:
                # Only extend with values that are iterable (list/dict) and not None
                for value in node.values():
                    if value is not None and isinstance(value, (dict, list)):
                        if isinstance(value, dict):
                            stack.append(value)
                        elif isinstance(value, list):
                            stack.extend(value)
        elif isinstance(node, list):
            # Handle case where node itself is a list
            stack.extend(node)
        
        # If we find an entry with entryId, yield it
        if isinstance(node, dict) and "entryId" in node:
            yield node


# ---------------------------------------------------------------------------
# Tweet unwrapping helper (NEW)
# ---------------------------------------------------------------------------
def _unwrap_tweet(raw: dict) -> Optional[dict]:
    """
    Accept anything returned by twitter-api-client and return the inner Tweet.
    Handles:
      • TweetResultsByRestIds      (scraper.tweets_by_ids / tweets_details)
      • TweetDetail               (thread endpoint)
      • Already-unwrapped tweets  (home timeline, etc.)
    """
    if not isinstance(raw, dict):
        return None
    if raw.get('rest_id'):                          # already a Tweet
        return raw

    data = raw.get('data', raw)                     # GraphQL payload root
    # common wrappers
    for key in (
        'tweetResult', 'tweet',                     # TweetDetail
        *(k for k in data if k.startswith('tweetResultByRestId'))  # batched
    ):
        maybe = data.get(key)
        if isinstance(maybe, dict):
            res = maybe.get('result') or maybe      # some variants skip 'result'
            if isinstance(res, dict) and res.get('rest_id'):
                return res
    return None


# ---------------------------------------------------------------------------
# AI Safety filtering helpers
# ---------------------------------------------------------------------------
def call_gpt_api(prompt: str,
                 model: str = "gpt-4.1",
                 temperature: float = 0.1) -> str:
    """
    Sends a single-prompt exchange to OpenAI ChatCompletions and
    returns the content string. Raises exceptions on failure.
    """
    if not openai_client:
        raise ValueError("OpenAI client not initialized")

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant specialized in AI safety content analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            raise ValueError("OpenAI API returned no content or unexpected response structure")
            
    except (OpenAI_APIError, OpenAI_RateLimitError) as e:
        logging.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during OpenAI API call: {e}", exc_info=True)
        raise


def is_ai_safety_tweet(content: str, author_username: str = "") -> bool:
    """
    Fast yes/no guard-rail: returns True iff the tweet's content
    is sufficiently focused on AI safety topics to be included in the feed.
    Uses OpenAI's GPT-4.1 model for fast classification.
    """
    if not openai_client:
        logging.warning("AI safety guard-rail check skipped: OpenAI client not initialized. Defaulting to True (fail-open).")
        return True

    # Clean the content - remove URLs, mentions, and extra whitespace
    cleaned_content = content
    if content:
        # Remove line separators that might be in Twitter text
        cleaned_content = content.replace('\u2028', '').replace('\u2029', '')
        # Strip extra whitespace
        cleaned_content = ' '.join(cleaned_content.split())
    
    if not cleaned_content or len(cleaned_content.strip()) < 10:
        logging.debug(f"Guard-rail: Tweet too short or empty, accepting by default")
        return True

    # Prepare content for analysis (truncate very long tweets)
    analysis_content = cleaned_content[:400]  # Twitter's max is 280, but threads can be longer
    analysis_text = f"Author: @{author_username}\n\nTweet: {analysis_content}"

    prompt = f"""
Analyze this tweet to determine if it is sufficiently focused on AI safety topics to be included in an AI safety content feed.

Content to analyze:
{analysis_text}

Rules for inclusion:
1. The content must substantially discuss AI safety, AI alignment, AI governance, AI policy, AI existential risk, or AI ethics.
2. Very general AI/ML technical content is NOT sufficient - there must be a clear safety/ethics/governance/risk angle.
3. Brief mentions of AI safety in otherwise unrelated content is NOT sufficient.
4. News about AI companies is only relevant if it has clear safety/governance implications.
5. Memes, jokes, or commentary about AI safety topics ARE acceptable if they demonstrate engagement with safety concepts.
6. Academic research posts about AI safety, alignment, or governance are relevant.
7. Policy discussions, regulatory developments, and governance frameworks for AI are relevant.

Respond with ONLY "yes" or "no". Use "yes" if you are reasonably confident the content meets the criteria, otherwise use "no".
"""
    
    try:
        response = call_gpt_api(
            prompt=prompt,
            model="gpt-4.1-mini",  # Use fast model for this guardrail
            temperature=0.1   # Low temperature for consistent yes/no
        )
        
        # Clean and validate response
        response = response.strip().lower()
        is_relevant = response == "yes"

        # Log the decision with truncated content for privacy
        content_preview = cleaned_content[:50] + "..." if len(cleaned_content) > 50 else cleaned_content
        if is_relevant:
            logging.debug(f"Guard-rail: Tweet by @{author_username} accepted: '{content_preview}'")
        else:
            logging.info(f"Guard-rail: Tweet by @{author_username} rejected: '{content_preview}'")

        return is_relevant
        
    except Exception as e:
        logging.warning(f"AI safety guard-rail check failed for @{author_username}: {type(e).__name__}: {e}. Defaulting to True (fail-open).")
        return True


# ---------------------------------------------------------------------------
# Database helper (kept from original)
# ---------------------------------------------------------------------------
def ensure_skipped_tweets_table():
    """Ensure the skipped_tweets table exists for tracking filtered content."""
    DB_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL")
    conn = psycopg2.connect(DB_URL)
    
    with conn, conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS skipped_tweets (
                tweet_id BIGINT PRIMARY KEY,
                author_username TEXT,
                content TEXT,
                like_count INTEGER DEFAULT 0,
                skipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_checked TIMESTAMP,
                reason TEXT DEFAULT 'ai_safety_filter',
                check_count INTEGER DEFAULT 0
            )
        ''')
    
    conn.close()


def record_skipped_tweet(tweet_id: int, author_username: str, content: str, like_count: int = 0, reason: str = 'ai_safety_filter'):
    """Record a tweet that was skipped due to filtering."""
    try:
        DB_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL")
        conn = psycopg2.connect(DB_URL)
        
        with conn, conn.cursor() as cur:
            cur.execute('''
                INSERT INTO skipped_tweets (tweet_id, author_username, content, like_count, reason, check_count)
                VALUES (%s, %s, %s, %s, %s, 0)
                ON CONFLICT (tweet_id) DO UPDATE SET
                    like_count = EXCLUDED.like_count,
                    reason = EXCLUDED.reason
            ''', (tweet_id, author_username, content[:500], like_count, reason))  # Truncate content to 500 chars
        
        conn.close()
        logging.debug(f"Recorded skipped tweet {tweet_id} by @{author_username} (likes: {like_count})")
    except Exception as e:
        logging.warning(f"Failed to record skipped tweet {tweet_id}: {e}")


def db_insert_many_by_thread(tweets: List[Dict[str, Any]], apply_ai_safety_filter: bool = True) -> Tuple[int, int]:
    """
    Insert/update tweets in PostgreSQL with thread-based AI safety filtering.
    Groups tweets by conversation/author and does one AI safety check per thread.
    If a thread passes, all tweets in the thread are inserted.
    If a thread fails, all tweets in the thread are added to skipped_tweets.
    Returns (inserted_count, skipped_count).
    """
    if not tweets:
        return 0, 0
    
    # Ensure skipped tweets table exists
    ensure_skipped_tweets_table()
    
    if not apply_ai_safety_filter or not openai_client:
        # No filtering - insert all tweets
        return _insert_tweets_to_db(tweets), 0
    
    # Group tweets by conversation_id and author
    threads = {}
    for tweet in tweets:
        key = (tweet['conversation_id'], tweet['author_username'])
        if key not in threads:
            threads[key] = []
        threads[key].append(tweet)
    
    approved_tweets = []
    rejected_tweets = []
    
    # Process each thread
    for (conversation_id, author_username), thread_tweets in threads.items():
        # Sort tweets by timestamp to get proper order for concatenation
        thread_tweets.sort(key=lambda t: t['timestamp'] or datetime.min.replace(tzinfo=timezone.utc))
        
        # Concatenate all content in the thread
        thread_content_parts = []
        for tweet in thread_tweets:
            if tweet['content']:
                thread_content_parts.append(tweet['content'].strip())
        
        thread_content = "\n---\n".join(thread_content_parts)
        
        # Do one AI safety check for the entire thread
        if is_ai_safety_tweet(thread_content, author_username):
            # Thread passes - add all tweets to approved
            approved_tweets.extend(thread_tweets)
            logging.info(f"THREAD APPROVED: {len(thread_tweets)} tweets by @{author_username} in conversation {conversation_id}")
        else:
            # Thread fails - add all tweets to rejected
            rejected_tweets.extend(thread_tweets)
            logging.info(f"THREAD REJECTED: {len(thread_tweets)} tweets by @{author_username} in conversation {conversation_id} - failed AI safety filter")
    
    # Record all rejected tweets in skipped_tweets table
    for tweet in rejected_tweets:
        record_skipped_tweet(
            tweet['tweet_id'], 
            tweet['author_username'], 
            tweet['content'],
            tweet['like_count'],
            'ai_safety_filter_thread'
        )
    
    # Insert approved tweets
    inserted_count = 0
    if approved_tweets:
        inserted_count = _insert_tweets_to_db(approved_tweets)
    
    skipped_count = len(rejected_tweets)
    
    logging.info(f"Thread-based filtering: {len(threads)} threads processed, {inserted_count} tweets inserted, {skipped_count} tweets skipped")
    return inserted_count, skipped_count


def _insert_tweets_to_db(tweets: List[Dict[str, Any]]) -> int:
    """
    Helper function to insert tweets into the database without filtering.
    Returns the number of tweets inserted.
    """
    if not tweets:
        return 0
    
    DB_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL")
    conn = psycopg2.connect(DB_URL)
    
    with conn, conn.cursor() as cur:
        rows = [(
            t['tweet_id'], 
            t['conversation_id'], 
            t['author_username'], 
            t['author_name'], 
            t['content'], 
            t['timestamp'],
            t['like_count'], 
            t['retweet_count'], 
            t['reply_count'], 
            t['quote_count'],
            t['media_urls'] or None, 
            t['external_links'] or None,
            t['url'],
            t.get('is_thread', False)
        ) for t in tweets]

        execute_values(cur, '''
            INSERT INTO tweets AS tweet_table (
                tweet_id, conversation_id, author_username, author_name, content, "timestamp",
                like_count, retweet_count, reply_count, quote_count, 
                media_urls, external_links, url, is_thread
            ) VALUES %s
            ON CONFLICT (tweet_id) DO UPDATE SET
                like_count    = EXCLUDED.like_count,
                retweet_count = EXCLUDED.retweet_count,
                reply_count   = EXCLUDED.reply_count,
                quote_count   = COALESCE(EXCLUDED.quote_count, tweet_table.quote_count),
                is_thread     = tweet_table.is_thread OR EXCLUDED.is_thread
        ''', rows)
        
        inserted_count = len(tweets)
    
    conn.close()
    return inserted_count


def db_insert_many(tweets: List[Dict[str, Any]], apply_ai_safety_filter: bool = True) -> Tuple[int, int]:
    """
    Insert/update tweets in PostgreSQL with upsert logic and AI safety filtering.
    Returns (inserted_count, skipped_count).
    """
    if not tweets:
        return 0, 0
    
    # Ensure skipped tweets table exists
    ensure_skipped_tweets_table()
    
    inserted_count = 0
    skipped_count = 0
    
    # Filter tweets through AI safety check if enabled
    filtered_tweets = []
    for tweet in tweets:
        if apply_ai_safety_filter and openai_client:
            if not is_ai_safety_tweet(tweet['content'], tweet['author_username']):
                # Record the skipped tweet
                record_skipped_tweet(
                    tweet['tweet_id'], 
                    tweet['author_username'], 
                    tweet['content'],
                    tweet['like_count'],
                    'ai_safety_filter'
                )
                skipped_count += 1
                logging.info(f"SKIP: Tweet {tweet['tweet_id']} by @{tweet['author_username']} failed AI safety filter")
                continue
        
        filtered_tweets.append(tweet)
    
    if not filtered_tweets:
        logging.info(f"No tweets passed AI safety filtering. {skipped_count} tweets were skipped.")
        return 0, skipped_count
    
    # Insert the filtered tweets
    inserted_count = _insert_tweets_to_db(filtered_tweets)
    
    logging.info(f"Successfully inserted {inserted_count} tweets, skipped {skipped_count} tweets")
    return inserted_count, skipped_count


# ---------------------------------------------------------------------------
# Tweet extraction helper (FIXED)
# ---------------------------------------------------------------------------
def extract_tweet_from_entry(entry: dict) -> Optional[dict]:
    """
    Extract tweet from timeline entry structure.
    Handles both HomeTimeline and HomeLatestTimeline formats, including modules.
    """
    try:
        # Handle TimelineTimelineModule first
        if "items" in entry.get("content", {}):
            for item in entry["content"]["items"]:
                t = extract_tweet_from_entry(item["item"])
                if t:
                    return t
        
        # Handle different entry structures
        if 'content' in entry:
            content = entry.get('content', {})
            
            # Regular TimelineTimelineItem
            if content.get('entryType') == 'TimelineTimelineItem':
                item_content = content.get('itemContent', {})
                tweet_results = item_content.get('tweet_results', {})
            else:
                # Try to get itemContent directly from content
                item_content = content.get('itemContent', {})
                tweet_results = item_content.get('tweet_results', {})
                
        elif 'item' in entry:
            # Alternative structure (from modules)
            item_content = entry.get('item', {}).get('itemContent', {})
            tweet_results = item_content.get('tweet_results', {})
        else:
            # Direct tweet_results
            tweet_results = entry.get('tweet_results', {})
        
        # Get the actual tweet result
        result = tweet_results.get('result', {})
        
        # Skip if not a tweet or if it's an ad
        if result.get('__typename') != 'Tweet':
            return None
            
        return result
    except Exception as e:
        logging.debug(f"Error extracting tweet: {e}")
        return None


def extract_tweets_from_module(entry: dict) -> List[dict]:
    """
    Extract tweets from a timeline module entry that contains multiple items.
    """
    tweets = []
    try:
        content = entry.get('content', {})
        if content.get('entryType') == 'TimelineTimelineModule':
            items = content.get('items', [])
            for item in items:
                tweet = extract_tweet_from_entry(item)
                if tweet:
                    tweets.append(tweet)
    except Exception as e:
        logging.debug(f"Error extracting tweets from module: {e}")
    
    return tweets


# ---------------------------------------------------------------------------
# Tweet transformation (FIXED)
# ---------------------------------------------------------------------------
def transform_tweet(tweet: Dict[str, Any], is_thread: bool = False) -> Optional[Dict[str, Any]]:
    """Transform GraphQL API tweet to our database format."""
    try:
        # Accept both raw tweets and still-wrapped objects
        tweet = _unwrap_tweet(tweet)
        if not tweet:
            return None
            
        # Extract tweet ID
        tweet_id = tweet.get('rest_id')
        if not tweet_id:
            logging.warning("No rest_id found in tweet")
            return None
            
        # Extract user data from core
        user_results = tweet.get('core', {}).get('user_results', {})
        user = user_results.get('result', {})
        
        # Get legacy data for both tweet and user
        tweet_legacy = tweet.get('legacy', {})
        user_legacy = user.get('legacy', {})
        
        if not user_legacy:
            logging.warning(f"No user legacy data for tweet {tweet_id}")
            return None
        
        # Parse timestamp
        timestamp_str = tweet_legacy.get('created_at')
        timestamp_dt = None
        if timestamp_str:
            try:
                timestamp_dt = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                logging.warning(f"Could not parse timestamp: {timestamp_str}")
                # Fallback to created_at_ms for non-English locales
                timestamp_ms = tweet_legacy.get('created_at_ms')
                if timestamp_ms:
                    try:
                        timestamp_dt = datetime.fromtimestamp(int(timestamp_ms)/1000, tz=timezone.utc)
                    except (ValueError, TypeError):
                        logging.warning(f"Could not parse timestamp_ms: {timestamp_ms}")
        else:
            # Try created_at_ms if created_at is missing
            timestamp_ms = tweet_legacy.get('created_at_ms')
            if timestamp_ms:
                try:
                    timestamp_dt = datetime.fromtimestamp(int(timestamp_ms)/1000, tz=timezone.utc)
                except (ValueError, TypeError):
                    logging.warning(f"Could not parse timestamp_ms: {timestamp_ms}")
        
        # Extract text content
        content = tweet_legacy.get('full_text', '')
        if not content:
            content = tweet_legacy.get('text', '')
        
        # Clean content
        if content:
            content = content.replace('\u2028', '').replace('\u2029', '')
        
        # Extract media URLs
        media_urls = []
        extended_entities = tweet_legacy.get('extended_entities', {})
        for media in extended_entities.get('media', []):
            if media.get('media_url_https'):
                media_urls.append(media['media_url_https'])
        
        # Extract external links
        external_links = []
        entities = tweet_legacy.get('entities', {})
        for url in entities.get('urls', []):
            if url.get('expanded_url'):
                external_links.append(url['expanded_url'])
        
        # Get conversation ID
        conversation_id = tweet_legacy.get('conversation_id_str', tweet_id)
        
        return {
            'tweet_id': int(tweet_id),
            'conversation_id': int(conversation_id),
            'author_username': user_legacy.get('screen_name', ''),
            'author_name': user_legacy.get('name', ''),
            'content': content,
            'timestamp': timestamp_dt,
            'like_count': tweet_legacy.get('favorite_count', 0),
            'retweet_count': tweet_legacy.get('retweet_count', 0),
            'reply_count': tweet_legacy.get('reply_count', 0),
            'quote_count': tweet_legacy.get('quote_count', 0),
            'media_urls': media_urls,
            'external_links': external_links,
            'url': f"https://x.com/i/web/status/{tweet_id}",
            'is_thread': is_thread
        }
        
    except Exception as e:
        logging.warning(f"Failed to transform tweet: {e}")
        return None


# ---------------------------------------------------------------------------
# Smart re-checking system for previously skipped tweets
# ---------------------------------------------------------------------------
def get_skipped_tweets_for_recheck(
    min_hours_since_check: int = 24,
    max_check_count: int = 3,
    min_like_threshold: int = 10
) -> List[Dict[str, Any]]:
    """
    Get tweets from skipped_tweets that are candidates for re-checking.
    
    Args:
        min_hours_since_check: Minimum hours since last check (or skip)
        max_check_count: Maximum times a tweet can be re-checked
        min_like_threshold: Only consider tweets that had some engagement when skipped
    
    Returns:
        List of tweet dictionaries with metadata for re-checking
    """
    try:
        DB_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL")
        conn = psycopg2.connect(DB_URL)
        
        with conn, conn.cursor() as cur:
            cur.execute('''
                SELECT tweet_id, author_username, content, like_count, 
                       skipped_at, last_checked, check_count, reason
                FROM skipped_tweets
                WHERE check_count < %s
                  AND like_count >= %s
                  AND reason IN ('ai_safety_filter', 'ai_safety_filter_thread')
                  AND (
                    last_checked IS NULL 
                    OR last_checked < NOW() - INTERVAL '%s hours'
                  )
                  AND skipped_at > NOW() - INTERVAL '7 days'  -- Only consider recent skips
                ORDER BY like_count DESC, skipped_at DESC
                LIMIT 50
            ''', (max_check_count, min_like_threshold, min_hours_since_check))
            
            columns = ['tweet_id', 'author_username', 'content', 'like_count', 
                      'skipped_at', 'last_checked', 'check_count', 'reason']
            
            results = []
            for row in cur.fetchall():
                tweet_data = dict(zip(columns, row))
                results.append(tweet_data)
            
        conn.close()
        return results
        
    except Exception as e:
        logging.error(f"Failed to get skipped tweets for recheck: {e}")
        return []


def update_skipped_tweet_check(tweet_id: int, new_like_count: int):
    """Update the check metadata for a previously skipped tweet."""
    try:
        DB_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL")
        conn = psycopg2.connect(DB_URL)
        
        with conn, conn.cursor() as cur:
            cur.execute('''
                UPDATE skipped_tweets 
                SET last_checked = NOW(),
                    check_count = check_count + 1,
                    like_count = %s
                WHERE tweet_id = %s
            ''', (new_like_count, tweet_id))
        
        conn.close()
        logging.debug(f"Updated check metadata for tweet {tweet_id} (likes: {new_like_count})")
        
    except Exception as e:
        logging.warning(f"Failed to update check metadata for tweet {tweet_id}: {e}")


def remove_from_skipped_tweets(tweet_id: int):
    """Remove a tweet from skipped_tweets (because it's now been accepted)."""
    try:
        DB_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL")
        conn = psycopg2.connect(DB_URL)
        
        with conn, conn.cursor() as cur:
            cur.execute('DELETE FROM skipped_tweets WHERE tweet_id = %s', (tweet_id,))
        
        conn.close()
        logging.debug(f"Removed tweet {tweet_id} from skipped_tweets")
        
    except Exception as e:
        logging.warning(f"Failed to remove tweet {tweet_id} from skipped_tweets: {e}")


def recheck_skipped_tweets(
    scraper: Scraper,
    popularity_threshold: int = 50,
    apply_ai_safety_filter: bool = True,
    delay_between_recheck_batches: float = 0.0
) -> Tuple[int, int]:
    """
    Re-check previously skipped tweets that may have gained popularity.
    
    Args:
        scraper: Twitter scraper instance
        popularity_threshold: Like count threshold for reconsidering tweets
        apply_ai_safety_filter: Whether to re-apply AI safety filtering
        delay_between_recheck_batches: Seconds to wait between batches of re-check fetches
    
    Returns:
        Tuple of (recovered_count, still_skipped_count)
    """
    logging.info("[*] Starting re-check of previously skipped tweets...")
    
    # Get candidates for re-checking
    candidates = get_skipped_tweets_for_recheck()
    if not candidates:
        logging.info("[*] No skipped tweets found for re-checking")
        return 0, 0
    
    logging.info(f"[*] Found {len(candidates)} candidate tweets for re-checking")
    
    # Batch fetch current tweet data
    tweet_ids = [str(c['tweet_id']) for c in candidates]
    recovered_tweets = []
    still_skipped = 0
    
    try:
        # Batch fetch current tweet data
        BATCH_SIZE = 100
        for i in range(0, len(tweet_ids), BATCH_SIZE):
            batch_ids = tweet_ids[i:i+BATCH_SIZE]
            
            # Add delay if this isn't the first batch AND a delay is set
            if i > 0 and delay_between_recheck_batches > 0:
                logging.info(f"Waiting {delay_between_recheck_batches}s before next re-check batch.")
                time.sleep(delay_between_recheck_batches)
            
            try:
                # Use tweets_by_ids for better rate limiting
                tweet_pages = scraper.tweets_by_ids(batch_ids)
                
                for page in tweet_pages:
                    for tweet_id_key, tweet_data in page.get('data', {}).items():
                        if not tweet_data:
                            continue
                            
                        # Find the corresponding candidate
                        tweet_id = int(tweet_id_key.replace('tweetResultByRestId_', ''))
                        candidate = next((c for c in candidates if c['tweet_id'] == tweet_id), None)
                        if not candidate:
                            continue
                        
                        # Unwrap and transform the tweet
                        unwrapped = _unwrap_tweet(tweet_data)
                        if not unwrapped:
                            update_skipped_tweet_check(tweet_id, candidate['like_count'])
                            still_skipped += 1
                            continue
                            
                        transformed = transform_tweet(unwrapped)
                        if not transformed:
                            update_skipped_tweet_check(tweet_id, candidate['like_count'])
                            still_skipped += 1
                            continue
                        
                        current_likes = transformed['like_count']
                        
                        # Check if tweet now meets popularity threshold
                        if current_likes >= popularity_threshold:
                            # Re-apply AI safety filter if enabled
                            if apply_ai_safety_filter and openai_client:
                                if not is_ai_safety_tweet(transformed['content'], transformed['author_username']):
                                    logging.info(f"RECHECK: Tweet {tweet_id} gained popularity ({current_likes} likes) but still fails AI safety filter")
                                    update_skipped_tweet_check(tweet_id, current_likes)
                                    still_skipped += 1
                                    continue
                            
                            # Tweet is now acceptable - add to recovery list
                            recovered_tweets.append(transformed)
                            logging.info(f"RECOVERED: Tweet {tweet_id} by @{transformed['author_username']} now has {current_likes} likes (was {candidate['like_count']})")
                            
                        else:
                            # Still below threshold, update check metadata
                            update_skipped_tweet_check(tweet_id, current_likes)
                            still_skipped += 1
                
            except Exception as e:
                logging.warning(f"Failed to fetch batch {batch_ids[:3]}...: {e}")
                # Update check metadata for failed fetches
                for batch_id in batch_ids:
                    candidate = next((c for c in candidates if c['tweet_id'] == int(batch_id)), None)
                    if candidate:
                        update_skipped_tweet_check(int(batch_id), candidate['like_count'])
                        still_skipped += 1
    
    except Exception as e:
        logging.error(f"Error during recheck process: {e}")
        return 0, 0
    
    # Insert recovered tweets
    if recovered_tweets:
        try:
            inserted_count, _ = db_insert_many(recovered_tweets, apply_ai_safety_filter=False)  # Already filtered
            
            # Remove successfully inserted tweets from skipped_tweets
            for tweet in recovered_tweets:
                remove_from_skipped_tweets(tweet['tweet_id'])
            
            logging.info(f"[✓] Recovered {inserted_count} previously skipped tweets")
            return inserted_count, still_skipped
            
        except Exception as e:
            logging.error(f"Failed to insert recovered tweets: {e}")
            return 0, still_skipped
    
    logging.info(f"[*] Re-check complete: 0 recovered, {still_skipped} still skipped")
    return 0, still_skipped


def get_skipped_tweets_stats() -> Dict[str, Any]:
    """Get statistics about skipped tweets for monitoring."""
    try:
        DB_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL")
        conn = psycopg2.connect(DB_URL)
        
        with conn, conn.cursor() as cur:
            # Get overall stats
            cur.execute('''
                SELECT 
                    COUNT(*) as total_skipped,
                    COUNT(CASE WHEN reason = 'ai_safety_filter' THEN 1 END) as ai_safety_skipped,
                    COUNT(CASE WHEN reason = 'ai_safety_filter_thread' THEN 1 END) as ai_safety_thread_skipped,
                    COUNT(CASE WHEN last_checked IS NOT NULL THEN 1 END) as rechecked_count,
                    AVG(like_count) as avg_like_count,
                    MAX(like_count) as max_like_count,
                    COUNT(CASE WHEN like_count >= 50 THEN 1 END) as high_engagement_skipped
                FROM skipped_tweets
                WHERE skipped_at > NOW() - INTERVAL '7 days'
            ''')
            
            stats = cur.fetchone()
            
            # Get recheck candidates (include both filter types)
            cur.execute('''
                SELECT COUNT(*) as recheck_candidates
                FROM skipped_tweets
                WHERE check_count < 3
                  AND like_count >= 10
                  AND reason IN ('ai_safety_filter', 'ai_safety_filter_thread')
                  AND (last_checked IS NULL OR last_checked < NOW() - INTERVAL '24 hours')
                  AND skipped_at > NOW() - INTERVAL '7 days'
            ''')
            
            recheck_stats = cur.fetchone()
            
        conn.close()
        
        return {
            'total_skipped_7d': stats[0] or 0,
            'ai_safety_skipped_7d': stats[1] or 0,
            'ai_safety_thread_skipped_7d': stats[2] or 0,
            'rechecked_count_7d': stats[3] or 0,
            'avg_like_count_7d': float(stats[4] or 0),
            'max_like_count_7d': stats[5] or 0,
            'high_engagement_skipped_7d': stats[6] or 0,
            'recheck_candidates': recheck_stats[0] or 0
        }
        
    except Exception as e:
        logging.error(f"Failed to get skipped tweets stats: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main (FIXED)
# ---------------------------------------------------------------------------
def main():
    # Load environment variables first
    load_dotenv()
    
    # Check database URL
    DB_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL")
    if not DB_URL:
        print("[FATAL] AI_SAFETY_TWEETS_DB_URL not set - abort")
        sys.exit(1)
    
    # Check OpenAI API key and initialize client
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    global openai_client
    openai_client = None
    
    if OPENAI_API_KEY:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("[INFO] OpenAI client initialized for AI safety filtering")
        except Exception as e:
            print(f"[WARNING] Failed to initialize OpenAI client: {e}. AI safety filtering will be disabled.")
            openai_client = None
    else:
        print("[WARNING] OPENAI_API_KEY not set - AI safety filtering will be disabled")
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Scrape Twitter Following timeline into Postgres"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=20, 
        help="Maximum tweets to fetch (default: 20)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save fetched tweets to JSON file for debugging"
    )
    parser.add_argument(
        "--threads",
        action="store_true",
        default=True,
        help="Fetch and process full threads for thread root tweets"
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=50,
        help="Maximum number of threads to process (default: 50)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--no-ai-filter",
        action="store_true",
        help="Disable AI safety filtering (insert all tweets)"
    )
    parser.add_argument(
        "--recheck-skipped",
        action="store_true",
        help="Re-check previously skipped tweets that may have gained popularity"
    )
    parser.add_argument(
        "--recheck-threshold",
        type=int,
        default=50,
        help="Like count threshold for reconsidering skipped tweets (default: 50)"
    )
    parser.add_argument(
        "--recheck-only",
        action="store_true",
        help="Only run re-checking, skip normal timeline scraping"
    )
    parser.add_argument(
        "--delay-after-timeline-fetch",
        type=float,
        default=10.0,
        help="Seconds to wait after fetching the main timeline before processing threads (default: 10.0)"
    )
    parser.add_argument(
        "--delay-between-thread-batches",
        type=float,
        default=15.0,
        help="Seconds to wait between batches of thread detail fetches (default: 15.0)"
    )
    parser.add_argument(
        "--delay-between-recheck-batches",
        type=float,
        default=15.0,
        help="Seconds to wait between batches of re-check fetches (default: 15.0)"
    )
    parser.add_argument(
        "--general-api-delay",
        type=float,
        default=3.0,
        help="A small general delay before major API call blocks (default: 3.0)"
    )
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Suppress verbose HTTP request logging from underlying libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Determine if AI safety filtering should be applied
    apply_ai_safety_filter = not args.no_ai_filter and openai_client is not None
    
    if apply_ai_safety_filter:
        logging.info("[*] AI safety filtering is ENABLED")
    else:
        logging.info("[*] AI safety filtering is DISABLED")
    
    # Display skipped tweets statistics
    if openai_client:
        try:
            stats = get_skipped_tweets_stats()
            if stats:
                individual_skipped = stats['ai_safety_skipped_7d']
                thread_skipped = stats['ai_safety_thread_skipped_7d']
                logging.info(f"[*] Skipped tweets stats (last 7 days): {stats['total_skipped_7d']} total "
                           f"({individual_skipped} individual, {thread_skipped} thread-based), "
                           f"{stats['high_engagement_skipped_7d']} high-engagement, "
                           f"{stats['recheck_candidates']} ready for recheck")
        except Exception as e:
            logging.debug(f"Could not retrieve skipped tweets stats: {e}")
    
    # Initialize Twitter account and scraper
    cookies_file = Path("twitter.cookies")
    account = Account(cookies=str(cookies_file))
    scraper = Scraper(cookies=str(cookies_file))   # same session for threads
    
    # Run re-checking of previously skipped tweets if requested
    total_recovered = 0
    total_still_skipped = 0
    
    if args.recheck_skipped or args.recheck_only:
        try:
            recovered, still_skipped = recheck_skipped_tweets(
                scraper=scraper,
                popularity_threshold=args.recheck_threshold,
                apply_ai_safety_filter=apply_ai_safety_filter,
                delay_between_recheck_batches=args.delay_between_recheck_batches
            )
            total_recovered += recovered
            total_still_skipped += still_skipped
            
            if args.recheck_only:
                logging.info(f"[✓] Re-check only mode complete! Recovered {total_recovered} tweets, {total_still_skipped} still skipped.")
                return
                
        except Exception as e:
            logging.error(f"[!] Re-checking failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    # Fetch timeline
    logging.info(f"[*] Fetching up to {args.limit} tweets from Following timeline...")
    
    # Add general delay before timeline fetch if not in recheck-only mode
    if args.general_api_delay > 0 and not args.recheck_only:
        logging.debug(f"General delay: Waiting {args.general_api_delay}s before timeline fetch.")
        time.sleep(args.general_api_delay)
    
    try:
        timeline = account.home_latest_timeline(limit=args.limit)
        
        # Wait after timeline fetch if specified and not in recheck-only mode
        if timeline and args.delay_after_timeline_fetch > 0 and not args.recheck_only:
            logging.info(f"Waiting {args.delay_after_timeline_fetch}s after timeline fetch...")
            time.sleep(args.delay_after_timeline_fetch)
        
        # Check what we got
        if not timeline:
            logging.error("[!] No timeline data returned")
            sys.exit(1)
            
        # Save raw timeline for debugging if requested
        if args.save_json:
            raw_timeline_path = Path("raw_timeline.json")
            try:
                with open(raw_timeline_path, "w", encoding="utf-8") as f:
                    json.dump(timeline, f, indent=2, ensure_ascii=False, default=str)
                logging.info(f"[+] Saved raw timeline to {raw_timeline_path}")
            except Exception as e:
                logging.error(f"[!] Failed to save raw timeline: {e}")
        
        # Process timeline entries
        all_tweets = []
        seen_tweet_ids = set()
        
        logging.info(f"[*] Processing {len(timeline)} timeline pages...")
        
        for page in timeline:                       # ← a GraphQL page
            for entry in iter_timeline_entries(page):
                eid = entry.get("entryId", "")
                if eid.startswith(("cursor-", "promoted-tweet")):   # ad / pagination
                    continue

                tweet = extract_tweet_from_entry(entry)
                if not tweet:
                    continue

                transformed = transform_tweet(tweet)
                if transformed and transformed["tweet_id"] not in seen_tweet_ids:
                    seen_tweet_ids.add(transformed["tweet_id"])
                    all_tweets.append(transformed)
                    logging.debug(f"Successfully processed tweet {transformed['tweet_id']}")
        
        logging.info(f"[*] Successfully transformed {len(all_tweets)} tweets")
        
        # Process threads if enabled
        if args.threads and all_tweets:
            logging.info("[*] Processing threads for thread root tweets...")
            # true roots = author == conversation starter and *not* a retweet
            # and (c) have at least 5 likes
            thread_roots = [
                t for t in all_tweets
                if (
                    t['tweet_id'] == t['conversation_id']          # conversation starter
                    and not t['content'].startswith('RT ')         # not a retweet
                    and t['like_count'] >= 5                       # ≥ 5 likes
                )
            ]
            
            # map ⇒ {conversation_id → author_username}
            root_authors = {t['tweet_id']: t['author_username'] for t in thread_roots}
            
            logging.info(f"[*] Found {len(thread_roots)} potential thread roots")
            
            all_thread_tweets = []
            
            # Limit thread processing
            max_threads_to_process = min(len(thread_roots), args.max_threads)
            
            # --- batch 220 ids at once (fat batch endpoint) ---------------
            BATCH = 220
            for i in range(0, max_threads_to_process, BATCH):
                batch_ids = [str(t['tweet_id']) for t in thread_roots[i:i+BATCH]]

                # Add delay if this isn't the first batch AND a delay is set
                if i > 0 and args.delay_between_thread_batches > 0:
                    logging.info(f"Waiting {args.delay_between_thread_batches}s before next thread batch.")
                    time.sleep(args.delay_between_thread_batches)

                logging.info(f"Fetching details for thread batch {i//BATCH + 1} (IDs: {batch_ids[:3]}...)")
                try:
                    # ➊ ask TweetDetails, not TweetResultsByRestIds
                    detail_pages = scraper.tweets_details(batch_ids)
                except Exception as e:
                    logging.warning(f"[thread] Batch failed {batch_ids[:3]}... : {e}")
                    continue

                for page in detail_pages:                       # TweetDetail -> pages
                    # walk every entry inside threaded_conversation_with_injections_v2
                    for entry in iter_timeline_entries(page):
                        t_raw = extract_tweet_from_entry(entry) or _unwrap_tweet(entry)
                        t_raw = _unwrap_tweet(t_raw)            # one more just in case
                        if not t_raw:
                            continue

                        # --- quick author filter  -------------------------------------------  # <<< NEW
                        conv_id = t_raw.get('legacy', {}).get('conversation_id_str') or t_raw.get('rest_id')
                        author  = (
                            t_raw.get('core', {})
                                 .get('user_results', {})
                                 .get('result', {})
                                 .get('legacy', {})
                                 .get('screen_name', '')
                        )
                        if author != root_authors.get(int(conv_id or 0)):
                            continue                                     # skip foreign replies

                        transformed = transform_tweet(t_raw, is_thread=True)
                        if transformed and transformed['tweet_id'] not in seen_tweet_ids:
                            seen_tweet_ids.add(transformed['tweet_id'])
                            all_thread_tweets.append(transformed)
            
            processed_threads = len(thread_roots[:max_threads_to_process])
            logging.info(f"[+] Processed {processed_threads} threads, found {len(all_thread_tweets)} additional tweets")
            all_tweets.extend(all_thread_tweets)
        
        # Save processed tweets if requested
        if args.save_json and all_tweets:
            output_path = Path("timeline_tweets.json")
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_tweets, f, indent=2, ensure_ascii=False, default=str)
                logging.info(f"[+] Saved processed tweets to {output_path}")
            except Exception as e:
                logging.error(f"[!] Failed to save JSON: {e}")
        
        # ──────────────────────────────────────────────────────────────────────────────
        # 3) final DB insert – keep every tweet that belongs to an approved thread
        # ──────────────────────────────────────────────────────────────────────────────
        if args.threads and 'root_authors' in locals():
            # Use thread-based filtering
            root_ids = set(root_authors.keys())                     # <<< NEW
            filtered_tweets = [
                t for t in all_tweets
                if (
                    # keep the root itself …
                    t['tweet_id'] in root_ids
                    # … or any tweet in that conversation **by the same author**
                    or (t['conversation_id'] in root_ids
                        and t['author_username'] == root_authors[t['conversation_id']])
                )
            ]
            logging.info(f"[*] Found {len(filtered_tweets)} tweets in approved threads")
        else:
            # Fallback to simple like count filtering
            filtered_tweets = [t for t in all_tweets if t.get('like_count', 0) >= 5]
            logging.info(f"[*] Found {len(filtered_tweets)} tweets with >= 5 likes")
        
        # Insert into database
        if filtered_tweets:
            logging.info("[*] Inserting into PostgreSQL...")
            try:
                # Use thread-based filtering when processing threads, individual filtering otherwise
                if args.threads and 'root_authors' in locals():
                    # Thread-based filtering: group by conversation and do one AI safety check per thread
                    inserted_count, skipped_count = db_insert_many_by_thread(filtered_tweets, apply_ai_safety_filter)
                else:
                    # Individual tweet filtering
                    inserted_count, skipped_count = db_insert_many(filtered_tweets, apply_ai_safety_filter)
                
                total_inserted = inserted_count + total_recovered
                total_skipped = skipped_count + total_still_skipped
                
                logging.info(f"[✓] Successfully inserted {inserted_count} new tweets, skipped {skipped_count} new tweets")
                if total_recovered > 0:
                    logging.info(f"[✓] Additionally recovered {total_recovered} previously skipped tweets")
                logging.info(f"[✓] Total session results: {total_inserted} tweets inserted, {total_skipped} tweets skipped")
            except Exception as e:
                logging.error(f"[!] Database error: {e}")
                sys.exit(1)
        else:
            if total_recovered > 0:
                logging.info(f"[✓] No new tweets to insert, but recovered {total_recovered} previously skipped tweets")
            else:
                logging.info("[*] No tweets to insert")
        
        logging.info("[✓] Done!")
        
    except Exception as e:
        logging.error(f"[!] Failed to fetch timeline: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()