#!/usr/bin/env python3
"""
Twitter Timeline Scraper for AI‑Safety Feed (MVP)
=================================================
Scrapes up to ~1000 tweets from the logged‑in "For You" timeline of a
 dedicated AI‑safety Twitter/X account, extracts rich metadata, optionally
expands author threads, and stores the data in PostgreSQL with upsert.

Requirements (install via pip):
    playwright>=1.44 psycopg2-binary python-dotenv

First‑time setup:
    1.  `pip install playwright psycopg2-binary python-dotenv
    2.  `playwright install chromium  # downloads browser binary
    3.  Run `python twitter_scraper.py --login once; a real browser
        window opens, log in manually, close the window.  This saves the
        authenticated storage state to `twitter_state.json.
    4.  Create the database table (DDL at bottom of this file) and set
        the environment variable `AI_SAFETY_TWEETS_DB_URL in *.env*.

Normal scraping run (cron/Airflow):
    $ python twitter_scraper.py

Environment variables (via .env or shell):
    AI_SAFETY_TWEETS_DB_URL  PostgreSQL connection string
    PLAYWRIGHT_STATE_PATH     (optional) path to storage‑state JSON file
                              [default: ./twitter_state.json]

This script is deliberately simple: synchronous Playwright API, single
browser context, sequential thread expansion.  Runtime for ~1000 tweets
with ~20 threads is typically < 90 seconds on a decent VPS.
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import logging

from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright, BrowserContext, Page, TimeoutError as PWTimeoutError
import psycopg2
from psycopg2.extras import execute_values, register_default_json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TWEET_TARGET: int = 100                # approx. tweets per run
SCROLL_PAUSE_RANGE = (1.5, 3.0)         # seconds – random to mimic humans
THREAD_EXPANSION_LIMIT = 10            # max # of timeline tweets whose threads we open
BROWSER_HEADLESS = True                 # set False for debugging
PLAYWRIGHT_STATE_FILE = Path(os.getenv("PLAYWRIGHT_STATE_PATH", "twitter_state.json"))

DB_URL = None  # loaded in main()

# Regex helpers
ID_RE = re.compile(r"/status/(\d+)")
HANDLE_RE = re.compile(r"@([A-Za-z0-9_]{1,15})")

# If you *store* any json columns, use:
# register_default_json(globally=True, loads=json.loads)

# ---------------------------------------------------------------------------
# NEW: Global variables for capturing network responses
# ---------------------------------------------------------------------------
home_chunks: list[dict] = []
detail_chunks: dict[str, dict] = {} # tweet_id (str) -> detail JSON

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def random_pause() -> None:
    """Sleep a random short interval to look less bot‑like."""
    time.sleep(random.uniform(*SCROLL_PAUSE_RANGE))


def ensure_storage_state(playwright: Playwright, force_login: bool = False) -> BrowserContext:
    """Create a browser context with saved session or (optionally) ask user to log in."""
    # Determine if an interactive login is required (either --login flag or missing state file)
    interactive_login_required = force_login or not PLAYWRIGHT_STATE_FILE.exists()

    # Launch the browser: non-headless for interactive login, otherwise respect BROWSER_HEADLESS
    # This ensures a visible browser window appears if the user needs to log in.
    actual_headless_state = False if interactive_login_required else BROWSER_HEADLESS
    browser = playwright.chromium.launch(headless=actual_headless_state, args=["--disable-blink-features=AutomationControlled"])

    if PLAYWRIGHT_STATE_FILE.exists() and not force_login:
        # If state file exists and we are not forcing login, use the saved state.
        # Browser was launched according to BROWSER_HEADLESS or False if state file was initially missing but now exists (edge case, covered by actual_headless_state logic).
        print(f"[*] Using existing login session from {PLAYWRIGHT_STATE_FILE!s}")
        context = browser.new_context(storage_state=str(PLAYWRIGHT_STATE_FILE))
        return context

    # Either --login flag used or missing state file: open interactive login.
    # Browser was launched with headless=False if interactive_login_required was true.
    print("[!] Opening a browser to perform (re)login · log in to twitter.com then close the window …")
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://twitter.com/login", timeout=0)
    print("[*] Waiting for user to finish login… (Ctrl+C to abort)")
    while True:
        try:
            # crude check: wait until we're redirected to /home and a tweet element appears
            print("[*] Checking for login completion (waiting for homepage tweet element)...")
            page.wait_for_selector('[data-testid="tweet"]', timeout=5_000)
            print("[+] Login completed by user.")
            break
        except PWTimeoutError:
            pass  # keep waiting
    # Save storage state
    context.storage_state(path=str(PLAYWRIGHT_STATE_FILE))
    print(f"[+] Storage state saved to {PLAYWRIGHT_STATE_FILE!s}")
    return context


# ---------------------------------------------------------------------------
# Core scraping logic
# ---------------------------------------------------------------------------

def scroll_timeline(page: Page, num_scrolls: int) -> None:
    """Scroll the home timeline a fixed number of times."""
    print(f"[*] Starting timeline scroll ({num_scrolls} times)...")
    for i in range(num_scrolls):
        page.evaluate("window.scrollBy(0, document.body.scrollHeight);")
        print(f"    Scroll {i+1}/{num_scrolls} completed.")
        random_pause()
    print("[+] Timeline scroll finished.")


# --- NEW: Network response capture ------------------------------------------

def _capture_response(resp):
    if resp.request.resource_type != "xhr" or not resp.ok:
        return
    url = resp.url
    if "HomeTimeline" in url:
        try:
            home_chunks.append(resp.json())
        except Exception as e:
            logging.warning(f"Failed to parse HomeTimeline JSON: {e} for URL: {url}")
    elif "TweetDetail" in url or "TweetResultByRestId" in url:
        try:
            data = resp.json()
            # focalTweetId query param appears only on TweetDetail calls initiated by clicking "Show this thread"
            # or by directly navigating to a tweet URL.
            # For TweetResultByRestId, the tweet ID is usually in the path or variables.
            # We need a robust way to get the ID of the main tweet this detail response is for.
            # The provided diff uses a regex on the URL, which is a good starting point for focalTweetId.
            m = re.search(r'focalTweetId":"(\d+)"', url) # From original user diff
            
            # If focalTweetId is not in URL, try to find it in the variables part of the GraphQL query
            # This is a common pattern for TweetDetail/TweetResultByRestId
            if not m and resp.request.post_data:
                try:
                    post_data_json = json.loads(resp.request.post_data)
                    variables = post_data_json.get("variables", {})
                    if isinstance(variables, str): # Sometimes variables is a JSON string
                        variables = json.loads(variables)
                    
                    focal_tweet_id_from_vars = variables.get("focalTweetId") or \
                                               variables.get("tweetId") or \
                                               variables.get("rest_id") # tweet_id might be called tweetId or rest_id
                    if focal_tweet_id_from_vars:
                        m = re.match(r"(\d+)", str(focal_tweet_id_from_vars)) # ensure it's digits
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse POST data for TweetDetail/ResultByRestId: {url}")


            tid = m.group(1) if m else None
            if tid:
                detail_chunks[tid] = data
            else:
                logging.warning(f"Could not extract focalTweetId for TweetDetail/ResultByRestId: {url}")
        except Exception as exc:
            logging.warning(f"Could not parse detail json for {url}: {exc}")


# --- NEW: JSON parsing helpers ----------------------------------------------

def _parse_tweet_result_json(tweet_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parses a 'tweet_results.result' (or similar path) JSON object from Twitter API
    into our standard tweet dictionary format.
    Handles different structures like 'tweet' for promoted tweets or missing fields.
    """
    if not tweet_result:
        return None

    # The actual tweet data can be under 'tweet' key (e.g., for "promoted tweet" items)
    # or directly at the root of tweet_result.
    tweet_data = tweet_result.get("tweet", tweet_result)

    legacy = tweet_data.get("legacy")
    core_user_results = tweet_data.get("core", {}).get("user_results", {}).get("result", {})
    
    if not legacy or not core_user_results or not core_user_results.get("legacy"):
        # logging.debug(f"Skipping item due to missing legacy or user data: {tweet_data.get('rest_id', 'N/A')}")
        return None # Essential data missing

    usr_legacy = core_user_results.get("legacy", {})
    tweet_id_str = tweet_data.get("rest_id")
    
    if not tweet_id_str or not tweet_id_str.isdigit():
        # logging.debug(f"Skipping item due to invalid or missing rest_id: {tweet_id_str}")
        return None # Essential tweet ID missing

    timestamp_str = legacy.get("created_at")
    timestamp_dt = None
    if timestamp_str:
        try:
            timestamp_dt = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %z %Y")
        except ValueError:
            logging.warning(f"Could not parse timestamp: {timestamp_str} for tweet {tweet_id_str}")
            timestamp_dt = None # Fallback or skip

    return {
        'tweet_id': int(tweet_id_str),
        'conversation_id': int(legacy.get("conversation_id_str", tweet_id_str)), # Fallback to tweet_id if not present
        'author_username': usr_legacy.get("screen_name"),
        'author_name': usr_legacy.get("name"),
        'content': legacy.get("full_text"),
        'timestamp': timestamp_dt,
        'like_count': legacy.get("favorite_count", 0),
        'retweet_count': legacy.get("retweet_count", 0),
        'reply_count': legacy.get("reply_count", 0),
        'quote_count': legacy.get("quote_count", 0), # Often 0 in timeline, updated from detail
        'media_urls': [m.get("media_url_https") for m in legacy.get("extended_entities", {}).get("media", []) if m.get("media_url_https")],
        'external_links': [u.get("expanded_url") for u in legacy.get("entities", {}).get("urls", []) if u.get("expanded_url")],
        'has_thread': False, # This might be harder to determine reliably from JSON alone without specific indicators.
                           # Can be approximated if tweet_id != conversation_id_str or if a "show thread" CTA exists in raw data.
                           # For now, let's simplify and rely on explicit thread expansion.
    }

def parse_home_chunk(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a single HomeTimeline JSON chunk."""
    parsed_tweets = []
    instructions = chunk.get("data", {}).get("home", {}).get("home_timeline_urt", {}).get("instructions", [])
    for instruction in instructions:
        if instruction.get("type") == "TimelineAddEntries":
            for entry in instruction.get("entries", []):
                if entry.get("entryId", "").startswith("tweet-"): # Standard tweets
                    content = entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result")
                    parsed_tweet = _parse_tweet_result_json(content)
                    if parsed_tweet:
                        parsed_tweets.append(parsed_tweet)
                elif entry.get("entryId", "").startswith("promoted-tweet-") or \
                     entry.get("entryId", "").startswith("promotedTweet"): # Promoted tweets
                    content = entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result")
                    # content might also be under entry.content.itemContent.promoted_tweet_results.result
                    if not content and entry.get("content", {}).get("itemContent", {}).get("promoted_tweet_results", {}):
                         content = entry.get("content", {}).get("itemContent", {}).get("promoted_tweet_results", {}).get("result")
                    parsed_tweet = _parse_tweet_result_json(content) # Promoted tweets have a 'tweet' sub-object
                    if parsed_tweet:
                        # logging.info(f"Parsed a promoted tweet: {parsed_tweet['tweet_id']}")
                        parsed_tweets.append(parsed_tweet) # Optionally skip these later
    return parsed_tweets


def parse_thread(detail_json: Dict[str, Any], root_tweet_id: int, root_author_username: str) -> List[Dict[str, Any]]:
    """
    Parse a TweetDetail JSON response to extract tweets belonging to the root author's thread.
    Excludes the root_tweet_id itself as it's already collected from HomeTimeline.
    """
    thread_tweets = []
    instructions = detail_json.get("data", {}).get("threaded_conversation_with_injections_v2", {}).get("instructions", [])
    if not instructions: # Fallback for different TweetDetail structures (e.g. TweetResultByRestId might have a different path)
        # Attempt to find entries in a structure similar to HomeTimeline for simplicity,
        # as TweetResultByRestId can have various forms.
        # This part may need adjustment based on actual observed JSON for TweetResultByRestId.
        results_container = detail_json.get("data", {}).get("tweetResult", {}).get("result", {}) # A common path for single tweet result
        if results_container and results_container.get("__typename") == "TweetWithVisibilityResults": # Ensure it's a tweet container
             results_container = results_container.get("tweet", {})

        if results_container and results_container.get("legacy"): # if it's a single tweet object itself
             # This case might not be a thread but a single tweet fetched.
             # This function expects a list of tweets in a thread.
             # For now, if it's just one tweet, we won't process it as a "thread".
             pass
        # A more common structure for threads might still use 'instructions' or 'entries' like HomeTimeline or older TweetDetail
        # If the primary 'threaded_conversation_with_injections_v2' path fails, we look for generic TimelineAddEntries
        elif detail_json.get("data"): # Check if data exists
            for key, value in detail_json["data"].items(): # Iterate through top-level keys in data
                if isinstance(value, dict) and "instructions" in value: # Find a dict with 'instructions'
                    instructions = value["instructions"]
                    break
    
    for instruction in instructions:
        if instruction.get("type") == "TimelineAddEntries":
            for entry in instruction.get("entries", []):
                # Tweets in a thread view can be nested differently
                content = None
                if entry.get("entryId", "").startswith(f"conversationthread-{root_tweet_id}-tweet-") or \
                   entry.get("entryId", "").startswith(f"tweet-{root_tweet_id}-tweet-") or \
                   entry.get("entryId", "").startswith("tweet-"): # Generic tweet entry in thread context
                    
                    item_content = entry.get("content", {}).get("itemContent", {})
                    if item_content and item_content.get("tweet_results", {}).get("result"):
                        content = item_content.get("tweet_results", {}).get("result")
                    # Sometimes it's nested under 'items' for conversation threads
                    elif entry.get("content", {}).get("items"): 
                        for item_entry in entry.get("content", {}).get("items", []):
                            if item_entry.get("entryId", "").endswith("-tweetDisplay"): # common suffix for tweets in items
                                item_content_inner = item_entry.get("item",{}).get("itemContent",{})
                                if item_content_inner and item_content_inner.get("tweet_results",{}).get("result"):
                                    content_candidate = item_content_inner.get("tweet_results",{}).get("result")
                                    # Check author and ensure it's not the root tweet again
                                    temp_parsed_tweet = _parse_tweet_result_json(content_candidate)
                                    if temp_parsed_tweet and \
                                       temp_parsed_tweet['author_username'] == root_author_username and \
                                       temp_parsed_tweet['tweet_id'] != root_tweet_id:
                                        thread_tweets.append(temp_parsed_tweet)
                                    # Processed, so continue to next entry
                        continue # Handled 'items' entry, move to next top-level entry

                if not content: # If not found through specific conversation thread paths or items
                    # Fallback to a more generic check if it's a direct tweet result within an entry
                    # (less common for threads but good for robustness with varying API responses)
                    if entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result"):
                         content = entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result")
                    # Check for a direct result if the entry itself is the tweet (e.g. in TweetResultByRestId)
                    elif entry.get("result",{}).get("legacy"): # If the entry is the tweet result itself
                        content = entry.get("result",{})


                if content:
                    parsed_tweet = _parse_tweet_result_json(content)
                    if parsed_tweet and \
                       parsed_tweet['author_username'] == root_author_username and \
                       parsed_tweet['tweet_id'] != root_tweet_id: # Exclude the root tweet itself
                        
                        # Update conversation_id for all thread tweets to be the root tweet's ID
                        parsed_tweet['conversation_id'] = root_tweet_id 
                        thread_tweets.append(parsed_tweet)
                        
                        # Additionally, if this thread tweet is the root_tweet_id, update its quote count from the detail view
                        # This is a bit redundant if parse_tweet_result_json already handles quote_count well,
                        # but detail views *sometimes* have more accurate quote counts for the focal tweet.
                        # However, the main purpose of parse_thread is to get *other* tweets in the thread.
                        # The root tweet's quote count update should ideally happen to the existing entry from HomeTimeline.
                        # For now, we rely on the quote_count from _parse_tweet_result_json.
                        
    # Deduplicate within the thread itself before returning, just in case
    if thread_tweets:
        thread_tweets = list({t['tweet_id']: t for t in thread_tweets}.values())
            
    return thread_tweets

# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------

def db_insert_many(tweets: Sequence[Dict[str, Any]]) -> None:
    if not tweets:
        return
    conn = psycopg2.connect(DB_URL)
    with conn, conn.cursor() as cur:
        rows = [(
            t['tweet_id'], t['conversation_id'], t['author_username'], t['author_name'], t['content'], t['timestamp'],
            t['like_count'], t['retweet_count'], t['reply_count'], t['quote_count'],
            t['media_urls'] or None, t['external_links'] or None
        ) for t in tweets]

        execute_values(cur, """
            INSERT INTO tweets (
                tweet_id, conversation_id, author_username, author_name, content, "timestamp",
                like_count, retweet_count, reply_count, quote_count, media_urls, external_links
            ) VALUES %s
            ON CONFLICT (tweet_id) DO UPDATE SET
                like_count      = EXCLUDED.like_count,
                retweet_count   = EXCLUDED.retweet_count,
                reply_count     = EXCLUDED.reply_count,
                quote_count     = COALESCE(EXCLUDED.quote_count, tweets.quote_count)
        """, rows)
    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global DB_URL
    load_dotenv()
    DB_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL")
    if not DB_URL:
        print("[FATAL] AI_SAFETY_TWEETS_DB_URL not set – abort")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Scrape Twitter timeline into Postgres")
    parser.add_argument("--login", action="store_true", help="Open interactive login to refresh session")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    with sync_playwright() as pw:
        context = ensure_storage_state(pw, force_login=args.login)
        page = context.new_page()

        # Attach response listener
        page.on("response", _capture_response)

        page.goto("https://twitter.com/home", timeout=0)
        try:
            # Wait for a timeline marker, not necessarily a full tweet,
            # as HomeTimeline API calls might populate faster.
            page.wait_for_selector('div[aria-label="Timeline: Your Home Timeline"]', timeout=20_000)
            logging.info("[*] Home timeline marker found.")
        except PWTimeoutError:
            logging.error("[!] Timeline marker not found – maybe not logged in or page structure changed? Use --login to refresh.")
            context.close()
            return

        logging.info("[*] Scrolling timeline to trigger HomeTimeline API calls...")
        scroll_timeline(page, 12)  # Increased scroll count

        # Wait an extra bit for last XHRs to complete after scrolling
        logging.info("[*] Waiting for final network responses...")
        page.wait_for_timeout(3000) # Increased from 1000ms to allow more time

        logging.info(f"[+] Captured {len(home_chunks)} HomeTimeline chunks and {len(detail_chunks)} TweetDetail chunks.")

        # Parse all HomeTimeline chunks
        timeline_tweets_unfiltered: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(home_chunks):
            # logging.info(f"[*] Parsing HomeTimeline chunk {idx+1}/{len(home_chunks)}")
            parsed_from_chunk = parse_home_chunk(chunk)
            timeline_tweets_unfiltered.extend(parsed_from_chunk)
            # logging.info(f"    Found {len(parsed_from_chunk)} tweets in chunk {idx+1}.")

        logging.info(f"[*] Parsed {len(timeline_tweets_unfiltered)} raw tweets from all HomeTimeline chunks.")

        # Deduplicate by tweet_id (keeping the first encountered version)
        # and filter out tweets without a valid tweet_id from parsing errors
        seen_tweet_ids = set()
        all_tweets_intermediate: List[Dict[str, Any]] = []
        for t in timeline_tweets_unfiltered:
            if t.get('tweet_id') and t['tweet_id'] not in seen_tweet_ids:
                all_tweets_intermediate.append(t)
                seen_tweet_ids.add(t['tweet_id'])
        
        # Merge accurate quote_count from detail view
        tweet_map = {t['tweet_id']: t for t in all_tweets_intermediate}
        for tid, detail in detail_chunks.items():
            root = detail.get('data', {}).get('tweetResult', {}).get('result', {})
            legacy = root.get('legacy', {})
            if legacy and (q := legacy.get('quote_count')):
                if int(tid) in tweet_map:
                    tweet_map[int(tid)]['quote_count'] = q
        all_tweets_intermediate = list(tweet_map.values())

        logging.info(f"[*] Found {len(all_tweets_intermediate)} unique tweets after initial parsing and deduplication from HomeTimeline.")

        # Thread expansion logic
        expanded_thread_tweets: List[Dict[str, Any]] = []
        tweets_for_thread_check = [t for t in all_tweets_intermediate if t.get('tweet_id') and t.get('author_username')] # Ensure necessary fields
        
        threads_to_expand_count = 0
        for td in tweets_for_thread_check:
            if threads_to_expand_count >= THREAD_EXPANSION_LIMIT:
                logging.info(f"[*] Reached thread expansion limit ({THREAD_EXPANSION_LIMIT}).")
                break

            # Check if it's potentially part of a thread (tweet_id != conversation_id)
            # OR if we already have detail_chunks for this tweet_id (might have been fetched e.g. by clicking "show replies")
            is_potential_thread_root = td['tweet_id'] != td.get('conversation_id', td['tweet_id'])
            has_detail_data = str(td['tweet_id']) in detail_chunks
            
            # Heuristic for 'has_thread' that was previously DOM-based:
            # Check if legacy.is_quote_status is false and if entities.urls is empty,
            # and if it's not a reply to someone else (conversation_id_str == id_str implies it's a root of its own convo)
            # This is complex to replicate perfectly from JSON alone without deeper inspection of API fields that indicate a "Show thread" button.
            # For now, we expand if tweet_id != conversation_id OR if we have detail_chunks.
            # The original plan suggested: `if td["tweet_id"] == td["conversation_id"]: continue` (typo, should be != for thread root)
            # Let's stick to: expand if it's a root of a conversation *that isn't just itself* or if we have details.

            if is_potential_thread_root or has_detail_data:
                 # Try to get actual detail JSON for this tweet to parse its thread
                tweet_id_str = str(td['tweet_id'])
                if tweet_id_str in detail_chunks:
                    logging.info(f"[*] Expanding thread for tweet {tweet_id_str} using pre-captured detail_chunk.")
                    thread_specific_tweets = parse_thread(detail_chunks[tweet_id_str], td['tweet_id'], td['author_username'])
                    if thread_specific_tweets:
                        expanded_thread_tweets.extend(thread_specific_tweets)
                        logging.info(f"    Added {len(thread_specific_tweets)} tweets from thread {tweet_id_str}.")
                    threads_to_expand_count += 1
                else:
                    # If not in detail_chunks, we could trigger a fetch, but the current design implies passive capture.
                    # The original plan didn't explicitly fetch if not in detail_chunks, it relied on what was captured.
                    # Let's assume for now that if it's not in detail_chunks, we don't expand unless is_potential_thread_root
                    # was true due to conversation_id mismatch (which implies it's part of a thread but not necessarily the root we want to expand FROM with a new fetch)
                    # The prompt's diff implies:
                    # if td["tweet_id"] == td["conversation_id"]: continue -> This would skip expanding tweets that are their own conversation root.
                    # This seems correct if we only want to expand *continuations* of threads seen on the timeline.
                    # Let's refine: only expand if detail_chunks has data for it.
                    # This means we only expand threads the user *might* have clicked on or that auto-loaded details.
                    # For a more proactive expansion of any tweet that *looks* like a thread head:
                    # if is_potential_thread_root and not has_detail_data:
                    # logging.info(f"[*] Tweet {tweet_id_str} is a potential thread root but no detail_chunk captured. Manual navigation would be needed.")
                    pass # Not fetching actively in this version based on user plan.
        
        if expanded_thread_tweets:
            logging.info(f"[*] Added {len(expanded_thread_tweets)} total tweets from thread expansions.")
            for t_expanded in expanded_thread_tweets:
                if t_expanded.get('tweet_id') and t_expanded['tweet_id'] not in seen_tweet_ids:
                    all_tweets_intermediate.append(t_expanded)
                    seen_tweet_ids.add(t_expanded['tweet_id'])

        # Final list of tweets after potential thread expansion and deduplication
        all_tweets_processed = list({t['tweet_id']: t for t in all_tweets_intermediate if t.get('tweet_id')}.values())
        logging.info(f"[*] Total unique tweets after thread expansion and final deduplication: {len(all_tweets_processed)}")
        
        # Save all parsed tweet data to a JSON file for inspection
        raw_output_path = Path("parsed_tweets_raw_graphql.json") # New name for GraphQL version
        try:
            with open(raw_output_path, "w", encoding="utf-8") as f:
                json.dump(all_tweets_processed, f, indent=2, ensure_ascii=False, default=str) # Use default=str for datetime
            logging.info(f"[+] Saved all {len(all_tweets_processed)} processed tweet objects to {raw_output_path!s}")
        except Exception as e:
            logging.error(f"[!] Error saving processed tweets to JSON: {e}")

        # Filter for tweets with more than 1 like (as per original logic)
        final_tweets_to_save: List[Dict[str, Any]] = []
        for td_final in all_tweets_processed:
            if td_final.get('like_count', 0) > 1:
                final_tweets_to_save.append(td_final)

        logging.info(f"[*] Found {len(final_tweets_to_save)} tweets with > 1 like after filtering for DB.")

        # --- Insert into DB --------------------------------------------------
        if final_tweets_to_save:
            logging.info("[*] Inserting into Postgres …")
            db_insert_many(final_tweets_to_save) # db_insert_many remains compatible
            logging.info(f"[✓] Successfully inserted/updated {len(final_tweets_to_save)} tweets.")
        else:
            logging.info("[*] No tweets to insert into Postgres.")
            
        logging.info("[✓] Done!")

        context.close()


if __name__ == "__main__":
    main()