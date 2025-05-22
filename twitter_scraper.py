#!/usr/bin/env python3
"""
Twitter Timeline Scraper for AI‑Safety Feed (MVP)
=================================================
Scrapes up to ~1000 tweets from the logged‑in "Following (chronological)" timeline of a
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
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import logging

from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright, BrowserContext, Page, TimeoutError as PWTimeoutError
import psycopg2
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCROLL_PAUSE_RANGE = (2.6, 6.1)         # seconds – random to mimic humans
THREAD_EXPANSION_LIMIT = 100            # max # of timeline tweets whose threads we open
BROWSER_HEADLESS = True                 # set False for debugging
PLAYWRIGHT_STATE_FILE = Path(os.getenv("PLAYWRIGHT_STATE_PATH", "twitter_state.json"))

DB_URL = None  # loaded in main()

# --- NEW: GraphQL API Constants for Tweet Detail Fetching ---
# WARNING: These values (especially Query ID and Features) can change with Twitter updates.
# They may need to be periodically verified and updated.
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
# Using TweetResultByRestId for fetching tweet details and replies.
# Query ID for TweetResultByRestId (example, verify from network requests if issues arise)
# Common IDs: 0hWvD_eA4fBnR72kWA_X3A, GazOgl_O_Kj0Y244X6gV1A, Q4M330Yn8OXPv2ZDPP8gvw etc.
TWEET_DETAIL_QUERY_ID = "0hWvD_eA4fBnR72kWA_X3A" 
TWEET_DETAIL_OPERATION_NAME = "TweetResultByRestId" # This operation usually pairs with such Query IDs
GRAPHQL_API_URL = f"https://twitter.com/i/api/graphql/{TWEET_DETAIL_QUERY_ID}/{TWEET_DETAIL_OPERATION_NAME}"

# Standard features often sent with GraphQL requests. This is a known common set.
DEFAULT_GRAPHQL_FEATURES = {
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "tweetypie_unmention_optimization_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "responsive_web_twitter_article_tweet_consumption_enabled": False,
    "tweet_awards_web_tipping_enabled": False,
    "responsive_web_home_pinned_timelines_enabled": True,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_action_policy_enabled": True,
    "responsive_web_media_download_video_enabled": False,
    "longform_notetweets_rich_text_read_enabled": True,
    "longform_notetweets_inline_media_enabled": True,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "responsive_web_live_event_timeline_management_enabled": True,
    "vibe_api_enabled": True,
    "responsive_web_text_conversations_enabled": False,
    "blue_business_profile_image_shape_enabled": True,
    "interactive_text_enabled": True,
    "responsive_web_enhance_cards_enabled": False
}

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
    
    # Clear cookies only when doing a fresh login to avoid hitting Twitter's cookie limit
    context.clear_cookies()
    
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
    """Scroll the timeline a fixed number of times without violating CSP."""
    logging.info(f"[*] Starting timeline scroll ({num_scrolls} times)...")
    for i in range(num_scrolls):
        page.mouse.wheel(0, 6_000) # Use mouse wheel for scrolling
        random_pause() # Call random_pause after each scroll
        logging.info("    Scroll %d/%d completed.", i + 1, num_scrolls)
    logging.info("[+] Timeline scroll finished.")


# --- NEW: Network response capture ------------------------------------------

def _capture_response(resp):
    if resp.request.resource_type not in ("xhr", "fetch") or not resp.ok:
        return
    url = resp.url
    if "HomeLatestTimeline" in url or "HomeTimeline" in url:   # safer
        try:
            home_chunks.append(resp.json())
        except Exception as e:
            logging.warning(f"Failed to parse HomeTimeline JSON: {e} for URL: {url}")
    elif "TweetDetail" in url or "TweetResultByRestId" in url:
        try:
            data = resp.json()

            m = re.search(r'[?&](?:focalTweetId|tweetId|rest_id)=(\d+)', url) # From original user diff
            
            # NEW: Fallback to query string parameters if regex fails
            if not m:
                parsed_url = urlparse(url)
                query_params = parse_qs(parsed_url.query)
                focal_tweet_id_from_query = query_params.get("focalTweetId", [None])[0] or \
                                            query_params.get("tweetId", [None])[0] or \
                                            query_params.get("rest_id", [None])[0]
                if focal_tweet_id_from_query:
                    m = re.match(r"(\d+)", str(focal_tweet_id_from_query)) # ensure it's digits

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
    
    usr_legacy = core_user_results.get("legacy")

    # Abort early if mandatory author fields are absent
    if (
        not legacy
        or not usr_legacy
        or not usr_legacy.get("screen_name")
        or not usr_legacy.get("name")
    ):
        return None   # skip ads, tombstones, withheld tweets

    usr_legacy = core_user_results.get("legacy", {})
    tweet_id_str = tweet_data.get("rest_id")
    
    if not tweet_id_str or not tweet_id_str.isdigit():
        # logging.debug(f"Skipping item due to invalid or missing rest_id: {tweet_id_str}")
        return None # Essential tweet ID missing

    full_text_content = legacy.get("full_text")
    if full_text_content:
        full_text_content = full_text_content.replace('\u2028', '').replace('\u2029', '')

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
        'content': full_text_content,
        'timestamp': timestamp_dt,
        'like_count': legacy.get("favorite_count", 0),
        'retweet_count': legacy.get("retweet_count", 0),
        'reply_count': legacy.get("reply_count", 0),
        'quote_count': legacy.get("quote_count", 0), # Often 0 in timeline, updated from detail
        'media_urls': [m.get("media_url_https") for m in legacy.get("extended_entities", {}).get("media", []) if m.get("media_url_https")],
        'external_links': [u.get("expanded_url") for u in legacy.get("entities", {}).get("urls", []) if u.get("expanded_url")],
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
                    continue                      # drop ads ASAP
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
        results_container = detail_json.get("data", {}).get("tweetResult", {}).get("result", {}) # A common path for single tweet result
        if results_container and results_container.get("__typename") == "TweetWithVisibilityResults": # Ensure it's a tweet container
             results_container = results_container.get("tweet", {})

        if results_container and results_container.get("legacy"): # if it's a single tweet object itself
             pass

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
                    if entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result"):
                         content = entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result")
                    # Check for a direct result if the entry itself is the tweet (e.g. in TweetResultByRestId)
                    elif entry.get("result",{}).get("legacy"): # If the entry is the tweet result itself
                        content = entry.get("result",{})


                if content:
                    parsed_tweet = _parse_tweet_result_json(content)
                    if parsed_tweet and \
                       parsed_tweet['author_username'] == root_author_username and \
                       parsed_tweet['tweet_id'] != root_tweet_id and \
                       parsed_tweet.get('conversation_id') == root_tweet_id: # Check conversation_id matches root
                        
                        # The 'conversation_id' is already correct due to the check above and parsing logic.
                        thread_tweets.append(parsed_tweet)
                        
                        
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
            t['media_urls'] or None, t['external_links'] or None,
            f"https://x.com/i/web/status/{t['tweet_id']}",  # Added URL
            t.get('is_thread', False) # is_thread
        ) for t in tweets]

        execute_values(cur, '''
            INSERT INTO tweets (
                tweet_id, conversation_id, author_username, author_name, content, "timestamp",
                like_count, retweet_count, reply_count, quote_count, media_urls, external_links, url, is_thread
            ) VALUES %s
            ON CONFLICT (tweet_id) DO UPDATE SET
                like_count      = EXCLUDED.like_count,
                retweet_count   = EXCLUDED.retweet_count,
                reply_count     = EXCLUDED.reply_count,
                quote_count     = COALESCE(EXCLUDED.quote_count, tweets.quote_count),
                is_thread       = tweets.is_thread OR EXCLUDED.is_thread
        ''', rows)
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
        
        # Attach response listener to the context to capture responses from all pages
        context.on("response", _capture_response)
        
        page = context.new_page()

        page.goto("https://twitter.com/home", timeout=0)
        # NEW ─ click the "Following" tab
        try:
            page.get_by_role("tab", name="Following").click()
            # Wait to confirm the tab switch actually happened
            page.wait_for_selector('[role="tab"][aria-selected="true"] >> text=Following', timeout=10000)
            logging.info("[*] Clicked 'Following' tab using get_by_role and confirmed selection.")
        except Exception:
            logging.info("[*] get_by_role(\"tab\", name=\"Following\") failed, trying fallback click.")
            page.click('a[role="tab"]:has-text("Following")')
            # Wait to confirm the tab switch actually happened
            page.wait_for_selector('[role="tab"][aria-selected="true"] >> text=Following', timeout=10000)
            logging.info("[*] Clicked 'Following' tab using fallback selector and confirmed selection.")

        try:
            # Wait for a timeline marker, not necessarily a full tweet,
            # as HomeTimeline API calls might populate faster.
            # UPDATED: Wait until tweets from the Following feed have loaded
            page.wait_for_selector('div[aria-label^="Timeline:"] [data-testid="tweet"]',
                                   timeout=20_000)
            logging.info("[*] Following timeline marker found.")
        except PWTimeoutError:
            logging.error("[!] Timeline marker not found – maybe not logged in or page structure changed? Use --login to refresh.")
            context.close()
            return

        logging.info("[*] Scrolling timeline to trigger HomeTimeline API calls...")
        scroll_timeline(page, 5)

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

        # Final list of tweets after initial parsing and deduplication.
        # (The passive expansion logic above has been removed, so this list is based on HomeTimeline + detail_chunks quote_count merge only at this point)
        all_tweets_processed = list({t['tweet_id']: t for t in all_tweets_intermediate if t.get('tweet_id')}.values())
        logging.info(f"[*] Total unique tweets after initial processing (before active thread fetching): {len(all_tweets_processed)}")
        
        # ------------------------------------------------------------------
        # NEW: actively fetch full threads
        # ------------------------------------------------------------------
        # Corrected root detection: tweet_id == conversation_id.
        # Fallback to tweet_id if conversation_id is missing for some reason.
        roots = [t for t in all_tweets_processed
                 if t.get('tweet_id') and t['tweet_id'] == t.get('conversation_id', t['tweet_id'])]

        logging.info(f"[*] Identified {len(roots)} potential thread roots for active fetching.")

        current_tweet_map = {t['tweet_id']: t for t in all_tweets_processed if t.get('tweet_id')}

        def pull_thread(context: BrowserContext, tid: int, author_username: str) -> None:
            # Check if author_username is valid, otherwise skip
            if not author_username:
                logging.warning(f"Skipping pull_thread for {tid} due to missing author_username.")
                return

            # The global _capture_response will populate detail_chunks[str(tid)]
            # Ensure detail_chunks is cleared or handled appropriately if this tweet was already visited
            # For simplicity here, we assume _capture_response appends/overwrites as needed.
            # If a specific entry for this tid already exists, we might want to clear it
            # or ensure the new response overwrites it correctly in _capture_response.
            # Current _capture_response for TweetDetail overwrites based on tid.

            logging.info(f"    Fetching thread for root tweet ID: {tid} via GraphQL API.") # MODIFIED
            
            cookies = context.cookies()
            csrf_token = None
            for cookie in cookies:
                if cookie['name'] == 'ct0':
                    csrf_token = cookie['value']
                    break
            
            if not csrf_token:
                logging.error(f"    CSRF token (ct0) not found for tweet {tid}. Skipping thread fetch via API.")
                return

            headers = {
                "authorization": BEARER_TOKEN,
                "x-csrf-token": csrf_token,
                "content-type": "application/json", # Added for POST
            }

            # Variables for TweetResultByRestId
            graphql_variables = {
                "tweetId": str(tid),
                "withControl": True,
                "withCommunity": True,
                "includePromotedContent": False,
                "withVoice": True,
                "withBirdwatchNotes": True, # For community notes
                "withDownvotePerspective": False,
                "withQuickPromoteEligibilityTweetFields": False,
                "withReactions": True, # For reaction counts/details
                "withEditControl": True, # For edit history info
                "withEditTweetQuoteCount": True,
                "withTweetQuoteCount": True,
                "withObservedEdits":True,
                "withV2Timeline": True, # Crucial for new timeline structures in replies
                "withSuperFollowsTweetFields": True,
                "withTrustedFriends": False,
                "withArticleRichContent": False,
                "withTextConversation": False, # Added for conversation context
                "withUserResults": True, # To get user objects in results
                "withClientEventToken":False, # Usually false
                "withIsQuoteTweet":True, # To identify quote tweets
            }
            
            # NEW: body for POST request
            body_payload = {
                "variables": graphql_variables,
                "features": DEFAULT_GRAPHQL_FEATURES,
                "fieldToggles": {"withArticleRichContentState": False} # As per user request
            }
            
            # Exponential backoff parameters
            max_retries = 3
            initial_delay = 5  # seconds
            current_retries = 0
            success = False

            while current_retries < max_retries:
                try:
                    logging.debug(f"Attempting GraphQL call for {tid} (Attempt {current_retries + 1}/{max_retries})")
                    # MODIFIED: Use post() with data=body_payload, remove params
                    api_response = context.request.post(
                        GRAPHQL_API_URL,
                        headers=headers,
                        data=json.dumps(body_payload), # Send as JSON string
                        timeout=20000 # 20s timeout
                    )
                    
                    # api_response.raise_for_status() # Playwright doesn't have this
                    # data = api_response.json()
                    if not api_response.ok:
                        logging.warning(
                            f"GraphQL call {tid} failed – HTTP {api_response.status}: "
                            f"{api_response.text()[:200]}" # Changed to use await for async method
                        )
                        # Trigger the retry loop by continuing, after logging and delay in finally
                        # Increment retries and sleep in the finally block before continuing
                        current_retries += 1
                        if current_retries < max_retries:
                             time.sleep(initial_delay)
                             initial_delay *= 2 # Exponential backoff
                        continue # Go to next iteration of while loop

                    data = api_response.json() # safe to call now, changed to use await
                    detail_chunks[str(tid)] = data # Store fetched data for parse_thread
                    
                    # Check if the detail_chunks were populated for this tid
                    if str(tid) in detail_chunks: # Data should be present if no error above
                        detail_json = detail_chunks[str(tid)]
                        thread_nodes = parse_thread(detail_json, tid, author_username)
                        
                        # Set is_thread for the root tweet
                        if tid in current_tweet_map:
                            current_tweet_map[tid]['is_thread'] = True
                        
                        newly_added_count = 0
                        for n in thread_nodes:
                            n['is_thread'] = True # Set is_thread for each thread node
                            if n.get('tweet_id') and n['tweet_id'] not in seen_tweet_ids:
                                all_tweets_processed.append(n) # This list is modified
                                seen_tweet_ids.add(n['tweet_id']) # This set is modified
                                current_tweet_map[n['tweet_id']] = n # Keep map updated
                                newly_added_count += 1
                        
                        if newly_added_count > 0:
                            logging.info(f"        Added {newly_added_count} new nodes from thread of {tid} (via GraphQL API).")
                        elif thread_nodes: # Even if no new nodes, if thread_nodes were found, it's a success
                            logging.info(f"        Processed thread for {tid} (via GraphQL API), no new nodes added beyond root.")
                        else: # No thread nodes found but API call was successful
                            logging.info(f"        API call for {tid} successful, but no thread content parsed or root already marked (via GraphQL API).")

                        success = True
                        break # Successfully fetched and processed
                    else:
                        # This path should ideally not be hit if data was assigned and no error prior.
                        logging.warning(f"    TweetDetail for {tid} not found in detail_chunks after API call. Retrying... (Attempt {current_retries + 1}/{max_retries})")

                except Exception as e: # Catches requests.exceptions.HTTPError, JSONDecodeError, PWTimeoutError (from request.get timeout) etc.
                    logging.warning(f"    GraphQL API call for tweet {tid} failed: {type(e).__name__} - {e}. Retrying in {initial_delay}s... (Attempt {current_retries + 1}/{max_retries})")
                
                finally: # Ensure delay happens if not successful and not last retry
                    if not success and current_retries < max_retries -1 : 
                         time.sleep(initial_delay)
                         initial_delay *= 2 # Exponential backoff

                current_retries += 1
            
            if not success:
                logging.error(f"    Max retries reached for tweet {tid} using GraphQL API. Giving up.")
            
            # p.close() is no longer needed as we are not creating a new page

        if roots:
            # Apply the THREAD_EXPANSION_LIMIT
            limited_roots = roots[:THREAD_EXPANSION_LIMIT]
            if len(roots) > THREAD_EXPANSION_LIMIT:
                logging.info(f"[*] Limiting thread expansion to {len(limited_roots)} roots (out of {len(roots)} potential roots, limit is {THREAD_EXPANSION_LIMIT}).")
            else:
                logging.info(f"[*] Preparing to expand {len(limited_roots)} thread roots (limit is {THREAD_EXPANSION_LIMIT}).")

            for i, r_tweet in enumerate(limited_roots, 1):
                # Ensure tweet_id and author_username are present
                root_tid = r_tweet.get('tweet_id')
                author = r_tweet.get('author_username')
                # Fallback for author_username from current_tweet_map if not directly in r_tweet (though it should be)
                if not author and root_tid in current_tweet_map:
                    author = current_tweet_map[root_tid].get('author_username')

                if root_tid and author:
                    delay = random.uniform(3.0, 6.0) # 3-6 s gap – slow but safe
                    logging.info(f"[{i}/{len(limited_roots)}] Sleeping {delay:.1f}s before thread fetch for {root_tid}")
                    time.sleep(delay)
                    pull_thread(context, root_tid, author) # runs in the main thread
                else:
                    logging.warning(f"Skipping root tweet for thread fetching due to missing ID or author: {r_tweet.get('tweet_id')}")
            
            logging.info("[*] Finished active thread fetching.")
            # After fetching, re-deduplicate all_tweets_processed as new nodes might have been added
            all_tweets_processed = list({t['tweet_id']: t for t in all_tweets_processed if t.get('tweet_id')}.values())
            logging.info(f"[*] Total unique tweets after active thread fetching and final deduplication: {len(all_tweets_processed)}")

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