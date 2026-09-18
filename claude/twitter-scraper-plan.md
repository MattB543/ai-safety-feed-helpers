# Bulk Twitter Profile Scraper: twscrape Implementation

## Overview

This document proposes a Python-based solution using the **twscrape** library to scrape 500+ public Twitter accounts, collecting their tweets, replies, quote tweets, and retweets, then hydrating full thread context for downstream LLM processing.

---

## Part 1: System Architecture

### Data Flow

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  bulk_profile_scraper   │ ──▶ │   hydrate_threads.py    │ ──▶ │   twitter-processer.py  │
│  (twscrape-based)       │     │   (twscrape or API.io)  │     │   (LLM-ready output)    │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
        │                               │                               │
        ▼                               ▼                               ▼
 scraped_tweets.jsonl             parents.json                  profiles_for_llm.txt
 (per-profile or merged)          (hydrated threads)
```

### What We're Collecting

For each of the 500 target profiles:

| Content Type     | twscrape Method             | Description                                |
| ---------------- | --------------------------- | ------------------------------------------ |
| **Tweets**       | `user_tweets()`             | Original posts by the user                 |
| **Replies**      | `user_tweets_and_replies()` | User's replies to others                   |
| **Quote Tweets** | Included in above           | User quoting another tweet with commentary |
| **Retweets**     | Included in above           | User sharing others' tweets                |

### Thread Hydration

When a scraped tweet is a reply or quote, we need the **parent context**:

```
Scraped Tweet (reply) ──references──▶ Parent Tweet (needs hydration)
                                            │
                                            ▼
                                      Grandparent Tweet (recursive)
```

---

## Part 2: twscrape Library Deep Dive

### What is twscrape?

**twscrape** is a Python library that interfaces with Twitter's internal GraphQL API (the same API the web app uses). It provides:

- **Account Pool Management**: Rotate multiple accounts to maximize throughput
- **Automatic Rate Limit Handling**: Switches accounts when one is rate-limited
- **Async/Await**: Parallel operations for speed
- **Parsed Data Models**: Clean Tweet/User objects with all fields

### Installation

```bash
pip install twscrape
```

### Core API Methods

```python
from twscrape import API, gather

api = API("accounts.db")

# ═══════════════════════════════════════════════════════════════
# USER LOOKUP
# ═══════════════════════════════════════════════════════════════

# Get user by username
user = await api.user_by_login("elonmusk")
print(user.id)           # 44196397
print(user.username)     # elonmusk
print(user.followersCount)

# Get user by numeric ID
user = await api.user_by_id(44196397)

# ═══════════════════════════════════════════════════════════════
# SCRAPING USER CONTENT
# ═══════════════════════════════════════════════════════════════

# User's tweets only (no replies)
# Returns: original tweets, retweets, quote tweets
# Limit: ~3200 tweets maximum per user (Twitter API limitation)
tweets = await gather(api.user_tweets(user.id, limit=500))

# User's tweets AND replies
# Returns: everything above + replies to other users
# This is what we want for comprehensive profile scraping
tweets = await gather(api.user_tweets_and_replies(user.id, limit=500))

# ═══════════════════════════════════════════════════════════════
# INDIVIDUAL TWEET DETAILS (for hydration)
# ═══════════════════════════════════════════════════════════════

# Get full tweet details by ID
tweet = await api.tweet_details(1234567890)

# Get replies to a specific tweet
replies = await gather(api.tweet_replies(1234567890, limit=100))

# ═══════════════════════════════════════════════════════════════
# SEARCH (alternative discovery method)
# ═══════════════════════════════════════════════════════════════

# Search tweets
tweets = await gather(api.search("from:elonmusk AI", limit=100))

# Search with filters
tweets = await gather(api.search("from:elonmusk since:2024-01-01", limit=100))
```

### Tweet Object Structure

```python
tweet.id                    # int: Tweet ID
tweet.id_str                # str: Tweet ID as string
tweet.rawContent            # str: Full tweet text
tweet.date                  # datetime: Posted timestamp
tweet.url                   # str: Tweet URL

# Author info
tweet.user.id               # int: User ID
tweet.user.username         # str: Handle (without @)
tweet.user.displayname      # str: Display name

# Engagement metrics
tweet.likeCount             # int
tweet.retweetCount          # int
tweet.replyCount            # int
tweet.quoteCount            # int
tweet.viewCount             # int (may be None)

# Relationship fields (CRITICAL for thread hydration)
tweet.inReplyToTweetId      # int | None: Parent tweet if this is a reply
tweet.inReplyToUser         # UserRef | None: Parent tweet author
tweet.quotedTweet           # Tweet | None: Full quoted tweet object
tweet.retweetedTweet        # Tweet | None: Full original tweet if RT
tweet.conversationId        # int: Thread conversation ID

# Media
tweet.media.photos          # List[Photo]: Image URLs
tweet.media.videos          # List[Video]: Video info
tweet.links                 # List[TextLink]: External URLs
```

### Tweet Type Detection

```python
def classify_tweet(tweet):
    """Determine the type of tweet."""
    if tweet.retweetedTweet:
        return "retweet"
    elif tweet.quotedTweet:
        return "quote_tweet"
    elif tweet.inReplyToTweetId:
        return "reply"
    else:
        return "original_tweet"
```

### Rate Limits & Constraints

| Constraint          | Value      | Notes                          |
| ------------------- | ---------- | ------------------------------ |
| Tweets per user     | ~3,200 max | Twitter API hard limit         |
| Rate limit window   | 15 minutes | Per endpoint, per account      |
| Requests per window | Varies     | ~150-900 depending on endpoint |
| Account switching   | Automatic  | twscrape handles this          |

**Throughput with multiple accounts:**

- 1 account: ~500-1000 tweets/hour
- 5 accounts: ~2500-5000 tweets/hour
- 10 accounts: ~5000-10000 tweets/hour

---

## Part 3: Account Setup

### Option 1: Cookie-Based Accounts (Recommended)

The most reliable method - uses existing logged-in sessions.

**Step 1: Export cookies from browser**

Using browser DevTools or a cookie export extension, get these cookies from twitter.com/x.com:

- `ct0` (CSRF token)
- `auth_token` (session token)

**Step 2: Create accounts file**

```
# accounts.txt
# Format: username:password:email:email_password:_:cookies
user1:pass1:user1@email.com:emailpass1:_:ct0=abc123;auth_token=xyz789
user2:pass2:user2@email.com:emailpass2:_:ct0=def456;auth_token=uvw012
```

**Step 3: Add accounts to twscrape**

```bash
twscrape add_accounts accounts.txt username:password:email:email_password:_:cookies
```

### Option 2: Login-Based Accounts

Let twscrape handle the login flow (less reliable due to Twitter's verification).

```bash
# Create file with just credentials
# accounts_login.txt
user1:pass1:user1@email.com:emailpass1
user2:pass2:user2@email.com:emailpass2

# Add accounts
twscrape add_accounts accounts_login.txt username:password:email:email_password

# Login (may require email verification)
twscrape login_accounts

# For manual email code entry
twscrape login_accounts --manual
```

### Managing Accounts

```bash
# View all accounts and their status
twscrape accounts

# Output:
# username  logged_in  active  last_used            total_req  error_msg
# user1     True       True    2024-01-15 10:30:00  1500       None
# user2     True       True    2024-01-15 10:25:00  1200       None
# user3     False      False   None                 0          Login error

# Re-login specific accounts
twscrape relogin user1 user2

# Re-login all failed accounts
twscrape relogin_failed

# Delete accounts
twscrape del_accounts user3
```

### Proxy Support

```python
# Per-account proxy
await api.pool.add_account("user", "pass", "email", "emailpass",
                           proxy="http://user:pass@proxy.com:8080")

# Global proxy
api = API(proxy="socks5://user:pass@127.0.0.1:1080")

# Environment variable
# TWS_PROXY=http://proxy.com:8080 python script.py
```

---

## Part 4: Implementation

### Script 1: `bulk_profile_scraper.py`

```python
#!/usr/bin/env python3
"""
bulk_profile_scraper.py

Scrape tweets, replies, quote tweets, and retweets from multiple public Twitter profiles.

Usage:
    # First, set up accounts
    twscrape add_accounts accounts.txt username:password:email:email_password:_:cookies

    # Then run scraper
    python bulk_profile_scraper.py --profiles profiles.txt --output scraped_data/

Requirements:
    pip install twscrape
"""

import asyncio
import json
import argparse
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict, Any, Optional
from contextlib import aclosing
from dataclasses import dataclass, asdict

from twscrape import API, gather
from twscrape.logger import set_log_level

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScraperConfig:
    tweets_per_profile: int = 500       # Target tweets per profile
    max_workers: int = 3                # Concurrent profile scrapers
    delay_min: float = 30.0             # Min delay between profiles (seconds)
    delay_max: float = 60.0             # Max delay between profiles (seconds)
    include_replies: bool = True        # Use user_tweets_and_replies vs user_tweets
    date_cutoff: Optional[datetime] = None  # Stop at tweets before this date
    db_path: str = "accounts.db"        # twscrape accounts database


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_tweet(tweet) -> str:
    """Determine the interaction type of a tweet."""
    if tweet.retweetedTweet:
        return "retweet"
    elif tweet.quotedTweet:
        return "quote_tweet"
    elif tweet.inReplyToTweetId:
        return "reply"
    else:
        return "tweet"


def extract_parent_ids(tweet) -> List[str]:
    """Extract all parent/referenced tweet IDs for hydration."""
    parent_ids = []

    # Reply parent
    if tweet.inReplyToTweetId:
        parent_ids.append(str(tweet.inReplyToTweetId))

    # Quoted tweet (twscrape often includes the full object, but we track ID anyway)
    if tweet.quotedTweet:
        parent_ids.append(str(tweet.quotedTweet.id))
        # Also get the quoted tweet's parent if it's a reply
        if tweet.quotedTweet.inReplyToTweetId:
            parent_ids.append(str(tweet.quotedTweet.inReplyToTweetId))

    # Retweeted tweet
    if tweet.retweetedTweet:
        parent_ids.append(str(tweet.retweetedTweet.id))
        # Also get the RT's parent if it's a reply
        if tweet.retweetedTweet.inReplyToTweetId:
            parent_ids.append(str(tweet.retweetedTweet.inReplyToTweetId))

    return list(set(parent_ids))  # Deduplicate


def tweet_to_dict(tweet, source_profile: str) -> Dict[str, Any]:
    """
    Convert twscrape Tweet object to dictionary format compatible with existing pipeline.

    This format matches what hydrate_parents_api.py and twitter-processer.py expect.
    """
    interaction_type = classify_tweet(tweet)
    parent_ids = extract_parent_ids(tweet)

    # Determine linked tweet ID based on interaction type
    linked_tweet_id = None
    if interaction_type == "reply":
        linked_tweet_id = str(tweet.inReplyToTweetId)
    elif interaction_type == "quote_tweet":
        linked_tweet_id = str(tweet.quotedTweet.id)
    elif interaction_type == "retweet":
        linked_tweet_id = str(tweet.retweetedTweet.id)

    # Extract URLs from links
    urls = []
    if tweet.links:
        urls = [link.url for link in tweet.links if link.url]

    # Extract media URLs
    media_urls = []
    if tweet.media:
        if tweet.media.photos:
            media_urls.extend([p.url for p in tweet.media.photos if p.url])
        if tweet.media.videos:
            media_urls.extend([v.thumbnailUrl for v in tweet.media.videos if v.thumbnailUrl])

    # Build the output dictionary
    result = {
        # Core identifiers
        "id": str(tweet.id),
        "conversation_id": str(tweet.conversationId) if tweet.conversationId else None,

        # Timestamps
        "created_at": tweet.date.strftime("%a %b %d %H:%M:%S %z %Y") if tweet.date else None,

        # Author info
        "screen_name": tweet.user.username if tweet.user else None,
        "author_id": str(tweet.user.id) if tweet.user else None,
        "author_display_name": tweet.user.displayname if tweet.user else None,

        # Content
        "text": tweet.rawContent or "",

        # Classification
        "interaction_type": interaction_type,
        "linked_tweet_id": linked_tweet_id,
        "parent_ids": parent_ids,

        # Reply metadata
        "reply_to_screen_name": tweet.inReplyToUser.username if tweet.inReplyToUser else None,
        "reply_to_user_id": str(tweet.inReplyToUser.id) if tweet.inReplyToUser else None,

        # Retweet metadata
        "retweeted_text": tweet.retweetedTweet.rawContent if tweet.retweetedTweet else None,
        "retweeted_screen_name": (
            tweet.retweetedTweet.user.username
            if tweet.retweetedTweet and tweet.retweetedTweet.user
            else None
        ),

        # Quote tweet metadata
        "quoted_text": tweet.quotedTweet.rawContent if tweet.quotedTweet else None,
        "quoted_screen_name": (
            tweet.quotedTweet.user.username
            if tweet.quotedTweet and tweet.quotedTweet.user
            else None
        ),

        # URLs and media
        "urls": urls,
        "media_urls": media_urls,

        # Engagement metrics
        "favorite_count": tweet.likeCount or 0,
        "retweet_count": tweet.retweetCount or 0,
        "reply_count": tweet.replyCount or 0,
        "quote_count": tweet.quoteCount or 0,
        "view_count": tweet.viewCount,

        # Scraping metadata
        "source_profile": source_profile,
        "scraped_at": datetime.utcnow().isoformat(),
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

async def get_user_id(api: API, username: str) -> Optional[int]:
    """Resolve username to numeric user ID."""
    try:
        user = await api.user_by_login(username)
        if user:
            print(f"  ✓ Resolved @{username} → ID {user.id}")
            return user.id
    except Exception as e:
        print(f"  ✗ Failed to resolve @{username}: {e}")
    return None


async def scrape_single_profile(
    api: API,
    username: str,
    config: ScraperConfig,
) -> List[Dict[str, Any]]:
    """
    Scrape all content from a single public profile.

    Collects: tweets, replies, quote tweets, retweets
    """
    print(f"\n{'─'*60}")
    print(f"📥 Scraping @{username}")
    print(f"{'─'*60}")

    # Resolve username to ID
    user_id = await get_user_id(api, username)
    if not user_id:
        return []

    tweets_collected: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    # Statistics
    stats = {"tweet": 0, "reply": 0, "quote_tweet": 0, "retweet": 0}

    try:
        # Choose endpoint based on config
        if config.include_replies:
            print(f"  📊 Using user_tweets_and_replies (limit={config.tweets_per_profile})")
            tweet_generator = api.user_tweets_and_replies(user_id, limit=config.tweets_per_profile)
        else:
            print(f"  📊 Using user_tweets (limit={config.tweets_per_profile})")
            tweet_generator = api.user_tweets(user_id, limit=config.tweets_per_profile)

        async with aclosing(tweet_generator) as gen:
            async for tweet in gen:
                tweet_id = str(tweet.id)

                # Skip duplicates
                if tweet_id in seen_ids:
                    continue
                seen_ids.add(tweet_id)

                # Optional date cutoff
                if config.date_cutoff and tweet.date:
                    if tweet.date < config.date_cutoff:
                        print(f"  📅 Hit date cutoff at {tweet.date.strftime('%Y-%m-%d')}")
                        break

                # Convert to dict and collect
                tweet_dict = tweet_to_dict(tweet, username)
                tweets_collected.append(tweet_dict)

                # Update stats
                stats[tweet_dict["interaction_type"]] += 1

                # Progress indicator every 100 tweets
                if len(tweets_collected) % 100 == 0:
                    print(f"  ... {len(tweets_collected)} tweets collected")

        # Summary
        print(f"\n  ✅ @{username} complete:")
        print(f"     Total: {len(tweets_collected)} tweets")
        print(f"     Breakdown: {stats['tweet']} original, {stats['reply']} replies, "
              f"{stats['quote_tweet']} quotes, {stats['retweet']} RTs")

    except Exception as e:
        print(f"  ❌ Error scraping @{username}: {e}")
        import traceback
        traceback.print_exc()

    return tweets_collected


async def worker(
    api: API,
    queue: asyncio.Queue,
    results: Dict[str, List[Dict[str, Any]]],
    worker_id: int,
    config: ScraperConfig,
):
    """Worker coroutine for parallel profile scraping."""
    while True:
        try:
            username = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        if username is None:  # Poison pill = shutdown
            break

        try:
            tweets = await scrape_single_profile(api, username, config)
            results[username] = tweets

            # Random delay between profiles to avoid rate limits
            delay = random.uniform(config.delay_min, config.delay_max)
            print(f"  ⏱️  Worker {worker_id}: waiting {delay:.1f}s before next profile...")
            await asyncio.sleep(delay)

        except Exception as e:
            print(f"  ❌ Worker {worker_id} fatal error on @{username}: {e}")
            results[username] = []
        finally:
            queue.task_done()


async def bulk_scrape(
    profiles: List[str],
    output_dir: Path,
    config: ScraperConfig,
) -> Dict[str, Any]:
    """
    Scrape multiple profiles in parallel.

    Args:
        profiles: List of usernames (without @)
        output_dir: Directory for output files
        config: Scraper configuration

    Returns:
        Statistics dictionary
    """
    print("\n" + "═"*70)
    print("🚀 BULK PROFILE SCRAPER")
    print("═"*70)
    print(f"📋 Profiles to scrape: {len(profiles)}")
    print(f"👷 Workers: {config.max_workers}")
    print(f"🎯 Tweets per profile: {config.tweets_per_profile}")
    print(f"⏱️  Delay between profiles: {config.delay_min}-{config.delay_max}s")
    print("═"*70)

    # Initialize API
    api = API(config.db_path)

    # Check account availability
    pool_stats = await api.pool.stats()
    print(f"\n📊 Account Pool Status:")
    print(f"   Total accounts: {pool_stats.get('total', 0)}")
    print(f"   Active accounts: {pool_stats.get('active', 0)}")

    if pool_stats.get('active', 0) == 0:
        print("\n❌ No active accounts! Set up accounts first:")
        print("   twscrape add_accounts accounts.txt username:password:email:email_password:_:cookies")
        return {"error": "No active accounts"}

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup queue and results storage
    queue: asyncio.Queue = asyncio.Queue()
    results: Dict[str, List[Dict[str, Any]]] = {}

    # Add profiles to queue
    for username in profiles:
        await queue.put(username)

    # Add poison pills for workers
    for _ in range(config.max_workers):
        await queue.put(None)

    # Start workers
    print(f"\n🏃 Starting {config.max_workers} workers...")
    workers = [
        asyncio.create_task(worker(api, queue, results, i + 1, config))
        for i in range(config.max_workers)
    ]

    # Wait for all work to complete
    await queue.join()
    await asyncio.gather(*workers)

    # ═══════════════════════════════════════════════════════════════════════════
    # WRITE OUTPUT FILES
    # ═══════════════════════════════════════════════════════════════════════════

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Main JSONL file with all tweets
    output_file = output_dir / f"scraped_tweets_{timestamp}.jsonl"

    total_tweets = 0
    all_parent_ids: Set[str] = set()
    all_tweet_ids: Set[str] = set()

    print(f"\n💾 Writing output files...")

    with output_file.open("w", encoding="utf-8") as f:
        for username, tweets in results.items():
            for tweet in tweets:
                f.write(json.dumps(tweet, ensure_ascii=False) + "\n")
                total_tweets += 1
                all_tweet_ids.add(tweet["id"])
                all_parent_ids.update(tweet.get("parent_ids", []))

    print(f"   ✓ {output_file.name}: {total_tweets} tweets")

    # Parent IDs file (for hydration step)
    # Only include IDs we don't already have
    parent_ids_to_hydrate = all_parent_ids - all_tweet_ids
    parent_ids_file = output_dir / f"parent_ids_to_hydrate_{timestamp}.txt"

    with parent_ids_file.open("w") as f:
        f.write("\n".join(sorted(parent_ids_to_hydrate)))

    print(f"   ✓ {parent_ids_file.name}: {len(parent_ids_to_hydrate)} IDs")

    # Profile summary file
    summary_file = output_dir / f"scrape_summary_{timestamp}.json"

    profile_stats = {}
    for username, tweets in results.items():
        if tweets:
            type_counts = {"tweet": 0, "reply": 0, "quote_tweet": 0, "retweet": 0}
            for t in tweets:
                type_counts[t["interaction_type"]] += 1
            profile_stats[username] = {
                "total": len(tweets),
                **type_counts
            }
        else:
            profile_stats[username] = {"total": 0, "error": True}

    summary = {
        "timestamp": timestamp,
        "config": asdict(config) if hasattr(config, '__dataclass_fields__') else {
            "tweets_per_profile": config.tweets_per_profile,
            "max_workers": config.max_workers,
            "include_replies": config.include_replies,
        },
        "totals": {
            "profiles_attempted": len(profiles),
            "profiles_with_data": sum(1 for t in results.values() if t),
            "profiles_failed": sum(1 for t in results.values() if not t),
            "total_tweets": total_tweets,
            "parent_ids_to_hydrate": len(parent_ids_to_hydrate),
        },
        "by_profile": profile_stats,
    }

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"   ✓ {summary_file.name}")

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "═"*70)
    print("📊 SCRAPING COMPLETE")
    print("═"*70)
    print(f"   Profiles scraped: {summary['totals']['profiles_with_data']}/{summary['totals']['profiles_attempted']}")
    print(f"   Total tweets: {summary['totals']['total_tweets']}")
    print(f"   Parent IDs to hydrate: {summary['totals']['parent_ids_to_hydrate']}")
    print(f"\n   Output directory: {output_dir}")
    print("═"*70)

    print("\n📋 Next steps:")
    print(f"   1. Hydrate parent tweets:")
    print(f"      python hydrate_threads.py --input {output_file} --parents {parent_ids_file}")
    print(f"   2. Process for LLM:")
    print(f"      python twitter-processer/twitter-processer.py --folder {output_dir}")

    return summary


def load_profiles(file_path: Path) -> List[str]:
    """Load profile usernames from file (one per line)."""
    profiles = []
    with file_path.open() as f:
        for line in f:
            # Strip whitespace and @ symbol
            username = line.strip().lstrip("@")
            # Skip empty lines and comments
            if username and not username.startswith("#"):
                profiles.append(username)
    return profiles


async def main():
    parser = argparse.ArgumentParser(
        description="Bulk scrape tweets from multiple Twitter profiles using twscrape",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python bulk_profile_scraper.py --profiles profiles.txt

  # Custom output and limits
  python bulk_profile_scraper.py --profiles profiles.txt --output ./data --limit 1000

  # More workers, faster scraping (requires more accounts)
  python bulk_profile_scraper.py --profiles profiles.txt --workers 5

  # Only original tweets (no replies)
  python bulk_profile_scraper.py --profiles profiles.txt --no-replies
        """
    )

    parser.add_argument(
        "--profiles", "-p",
        type=Path,
        required=True,
        help="File with usernames to scrape (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("scraped_data"),
        help="Output directory (default: scraped_data/)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=500,
        help="Max tweets per profile (default: 500)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="Number of concurrent workers (default: 3)"
    )
    parser.add_argument(
        "--no-replies",
        action="store_true",
        help="Only scrape original tweets (exclude replies)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="accounts.db",
        help="twscrape accounts database path (default: accounts.db)"
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=30.0,
        help="Minimum delay between profiles in seconds (default: 30)"
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=60.0,
        help="Maximum delay between profiles in seconds (default: 60)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        set_log_level("DEBUG")

    # Load profiles
    if not args.profiles.exists():
        print(f"❌ Profiles file not found: {args.profiles}")
        sys.exit(1)

    profiles = load_profiles(args.profiles)
    print(f"📋 Loaded {len(profiles)} profiles from {args.profiles}")

    if not profiles:
        print("❌ No profiles to scrape!")
        sys.exit(1)

    # Build config
    config = ScraperConfig(
        tweets_per_profile=args.limit,
        max_workers=args.workers,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        include_replies=not args.no_replies,
        db_path=args.db,
    )

    # Run scraper
    await bulk_scrape(profiles, args.output, config)


if __name__ == "__main__":
    asyncio.run(main())
```

### Script 2: `hydrate_threads.py`

```python
#!/usr/bin/env python3
"""
hydrate_threads.py

Fetch parent tweets to build complete thread context for scraped tweets.
Uses twscrape to fetch individual tweet details.

Usage:
    python hydrate_threads.py --input scraped_tweets.jsonl --parents parent_ids.txt

This creates parents.json which twitter-processer.py can use for thread context.
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import Set, Dict, Any, List, Optional
from datetime import datetime

from twscrape import API
from twscrape.logger import set_log_level

# Configuration
BATCH_SIZE = 50          # Progress reporting interval
MAX_DEPTH = 3            # Maximum thread depth to follow
DELAY_BETWEEN = 0.5      # Delay between tweet fetches (seconds)
DB_PATH = "accounts.db"


async def fetch_tweet(api: API, tweet_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single tweet by ID and convert to hydration format."""
    try:
        tweet = await api.tweet_details(tweet_id)
        if not tweet:
            return None

        return {
            "id": str(tweet.id),
            "text": tweet.rawContent or "",
            "created_at": tweet.date.isoformat() if tweet.date else None,
            "screen_name": tweet.user.username if tweet.user else None,
            "author_id": str(tweet.user.id) if tweet.user else None,

            # Relationship fields for recursive hydration
            "in_reply_to_status_id_str": str(tweet.inReplyToTweetId) if tweet.inReplyToTweetId else None,
            "in_reply_to_screen_name": tweet.inReplyToUser.username if tweet.inReplyToUser else None,
            "quoted_status_id_str": str(tweet.quotedTweet.id) if tweet.quotedTweet else None,

            # Include quoted tweet text if available (twscrape often has full object)
            "quoted_text": tweet.quotedTweet.rawContent if tweet.quotedTweet else None,
            "quoted_screen_name": (
                tweet.quotedTweet.user.username
                if tweet.quotedTweet and tweet.quotedTweet.user
                else None
            ),
        }

    except Exception as e:
        print(f"  ⚠️ Failed to fetch tweet {tweet_id}: {e}")
        return None


async def hydrate_tweets(
    api: API,
    tweet_ids: List[str],
    existing_ids: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Hydrate a list of tweet IDs.

    Returns:
        Dictionary mapping tweet ID to tweet data
    """
    results: Dict[str, Dict[str, Any]] = {}
    ids_to_fetch = [tid for tid in tweet_ids if tid not in existing_ids]

    if not ids_to_fetch:
        return results

    print(f"  🔄 Fetching {len(ids_to_fetch)} tweets...")

    for i, tweet_id in enumerate(ids_to_fetch):
        tweet_data = await fetch_tweet(api, int(tweet_id))

        if tweet_data:
            results[tweet_id] = tweet_data

        # Progress indicator
        if (i + 1) % BATCH_SIZE == 0:
            print(f"     ... {i + 1}/{len(ids_to_fetch)} processed")

        # Small delay to avoid rate limits
        await asyncio.sleep(DELAY_BETWEEN)

    print(f"  ✅ Fetched {len(results)}/{len(ids_to_fetch)} tweets")
    return results


def extract_parent_ids_from_hydrated(tweets: Dict[str, Dict[str, Any]]) -> Set[str]:
    """Extract parent IDs from hydrated tweets for recursive fetching."""
    parent_ids: Set[str] = set()

    for tweet_data in tweets.values():
        # Reply parent
        if tweet_data.get("in_reply_to_status_id_str"):
            parent_ids.add(tweet_data["in_reply_to_status_id_str"])

        # Quoted tweet
        if tweet_data.get("quoted_status_id_str"):
            parent_ids.add(tweet_data["quoted_status_id_str"])

    return parent_ids


async def hydrate_threads(
    input_file: Path,
    parent_ids_file: Optional[Path],
    output_file: Path,
    max_depth: int = MAX_DEPTH,
):
    """
    Recursively hydrate parent tweets to build complete thread context.

    Args:
        input_file: JSONL file with scraped tweets
        parent_ids_file: Optional file with pre-computed parent IDs
        output_file: Output JSON file for hydrated parents
        max_depth: Maximum recursion depth
    """
    print("\n" + "═"*70)
    print("🧵 THREAD HYDRATION")
    print("═"*70)

    # Collect existing tweet IDs and parent IDs from input
    existing_ids: Set[str] = set()
    parent_ids: Set[str] = set()

    print(f"📖 Loading scraped tweets from {input_file}...")

    with input_file.open() as f:
        for line in f:
            if not line.strip():
                continue
            tweet = json.loads(line)
            existing_ids.add(tweet["id"])
            parent_ids.update(tweet.get("parent_ids", []))

    print(f"   Found {len(existing_ids)} scraped tweets")
    print(f"   Found {len(parent_ids)} parent references")

    # Optionally load pre-computed parent IDs
    if parent_ids_file and parent_ids_file.exists():
        print(f"📖 Loading parent IDs from {parent_ids_file}...")
        with parent_ids_file.open() as f:
            for line in f:
                tid = line.strip()
                if tid:
                    parent_ids.add(tid)
        print(f"   Total parent IDs: {len(parent_ids)}")

    # Remove IDs we already have
    ids_to_hydrate = parent_ids - existing_ids
    print(f"\n📊 Need to hydrate: {len(ids_to_hydrate)} parent tweets")

    if not ids_to_hydrate:
        print("✅ All parent tweets already available!")
        # Write empty parents file for consistency
        with output_file.open("w") as f:
            json.dump({}, f)
        return

    # Initialize API
    api = API(DB_PATH)

    # Check accounts
    stats = await api.pool.stats()
    print(f"\n📊 Account pool: {stats.get('active', 0)} active accounts")

    if stats.get('active', 0) == 0:
        print("❌ No active accounts!")
        return

    # Recursive hydration
    all_hydrated: Dict[str, Dict[str, Any]] = {}
    current_ids = list(ids_to_hydrate)
    current_depth = 0

    while current_ids and current_depth < max_depth:
        current_depth += 1
        print(f"\n{'─'*60}")
        print(f"🔍 Depth {current_depth}/{max_depth}: {len(current_ids)} tweets to fetch")
        print(f"{'─'*60}")

        # Hydrate current batch
        new_tweets = await hydrate_tweets(
            api,
            current_ids,
            existing_ids | set(all_hydrated.keys())
        )

        all_hydrated.update(new_tweets)

        # Find new parent IDs from what we just fetched
        new_parent_ids = extract_parent_ids_from_hydrated(new_tweets)

        # Filter to only IDs we don't have yet
        already_have = existing_ids | set(all_hydrated.keys())
        next_ids = list(new_parent_ids - already_have)

        if next_ids:
            print(f"   🔗 Found {len(next_ids)} new parent IDs to fetch")

        current_ids = next_ids

    if current_ids:
        print(f"\n⚠️ Stopped at max depth {max_depth} with {len(current_ids)} unfetched parents")

    # Save results
    print(f"\n💾 Saving {len(all_hydrated)} hydrated tweets to {output_file}...")

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_hydrated, f, indent=2, ensure_ascii=False)

    print("\n" + "═"*70)
    print("✅ HYDRATION COMPLETE")
    print("═"*70)
    print(f"   Hydrated tweets: {len(all_hydrated)}")
    print(f"   Output file: {output_file}")
    print("═"*70)


async def main():
    parser = argparse.ArgumentParser(
        description="Hydrate parent tweets for thread context using twscrape"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input JSONL file from bulk scraper"
    )
    parser.add_argument(
        "--parents", "-p",
        type=Path,
        default=None,
        help="Optional file with parent IDs to hydrate (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("parents.json"),
        help="Output JSON file (default: parents.json)"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=MAX_DEPTH,
        help=f"Maximum thread depth to follow (default: {MAX_DEPTH})"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DB_PATH,
        help=f"twscrape accounts database (default: {DB_PATH})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        set_log_level("DEBUG")

    global DB_PATH
    DB_PATH = args.db

    await hydrate_threads(
        args.input,
        args.parents,
        args.output,
        args.depth,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Part 5: Complete Workflow

### Step 0: Prepare Account Pool

```bash
# Install twscrape
pip install twscrape

# Create accounts file with cookies
# accounts.txt format: username:password:email:email_password:_:cookies
echo "myaccount:mypass:me@email.com:emailpass:_:ct0=abc123;auth_token=xyz789" > accounts.txt

# Add to twscrape
twscrape add_accounts accounts.txt username:password:email:email_password:_:cookies

# Verify
twscrape accounts
```

### Step 1: Create Target Profile List

```bash
# profiles.txt - one username per line
cat > profiles.txt << 'EOF'
elonmusk
OpenAI
AnthropicAI
GoogleDeepMind
sama
karpathy
ylecun
# Add your 500 profiles here...
EOF
```

### Step 2: Run Bulk Scraper

```bash
# Basic run
python bulk_profile_scraper.py --profiles profiles.txt

# With custom settings
python bulk_profile_scraper.py \
    --profiles profiles.txt \
    --output ./scraped_data \
    --limit 500 \
    --workers 3 \
    --delay-min 30 \
    --delay-max 60
```

**Output files:**

- `scraped_data/scraped_tweets_YYYY-MM-DD-HH-MM-SS.jsonl` - All tweets
- `scraped_data/parent_ids_to_hydrate_*.txt` - IDs needing hydration
- `scraped_data/scrape_summary_*.json` - Statistics

### Step 3: Hydrate Thread Context

```bash
python hydrate_threads.py \
    --input scraped_data/scraped_tweets_*.jsonl \
    --parents scraped_data/parent_ids_to_hydrate_*.txt \
    --output scraped_data/parents.json \
    --depth 3
```

### Step 4: Process for LLM

```bash
# Use existing twitter-processer.py
python twitter-processer/twitter-processer.py --folder scraped_data/
```

**Final output:**

- `scraped_data/profiles_for_llm.txt` - LLM-ready text format

---

## Part 6: Estimated Performance

### Time Estimates for 500 Profiles

| Accounts | Workers | Tweets/Profile | Est. Time    |
| -------- | ------- | -------------- | ------------ |
| 1        | 1       | 500            | ~20-25 hours |
| 3        | 3       | 500            | ~7-10 hours  |
| 5        | 5       | 500            | ~4-6 hours   |
| 10       | 5       | 500            | ~3-4 hours   |

### Resource Requirements

- **Accounts**: 3-10 Twitter accounts with valid cookies
- **Storage**: ~100-500 MB for 500 profiles × 500 tweets
- **Memory**: ~500 MB - 1 GB during scraping
- **Network**: Stable internet connection

### Rate Limit Mitigation

The scripts include several safeguards:

1. **Random delays** between profiles (30-60s default)
2. **Account rotation** handled by twscrape
3. **Graceful error handling** - continues on individual failures
4. **Progress checkpointing** - can resume from summary file
5. **Configurable concurrency** - reduce workers if hitting limits

---

## Part 7: Data Format Reference

### Scraped Tweet Format (JSONL)

```json
{
  "id": "1234567890",
  "conversation_id": "1234567890",
  "created_at": "Mon Jan 15 12:30:45 +0000 2024",
  "screen_name": "username",
  "author_id": "9876543210",
  "author_display_name": "Display Name",
  "text": "Tweet content here...",
  "interaction_type": "reply",
  "linked_tweet_id": "1234567889",
  "parent_ids": ["1234567889", "1234567888"],
  "reply_to_screen_name": "other_user",
  "reply_to_user_id": "1111111111",
  "retweeted_text": null,
  "retweeted_screen_name": null,
  "quoted_text": null,
  "quoted_screen_name": null,
  "urls": ["https://example.com/article"],
  "media_urls": ["https://pbs.twimg.com/media/abc.jpg"],
  "favorite_count": 150,
  "retweet_count": 25,
  "reply_count": 10,
  "quote_count": 5,
  "view_count": 5000,
  "source_profile": "username",
  "scraped_at": "2024-01-15T12:45:00"
}
```

### Hydrated Parent Format (JSON)

```json
{
  "1234567889": {
    "id": "1234567889",
    "text": "Parent tweet content...",
    "created_at": "2024-01-15T12:00:00",
    "screen_name": "other_user",
    "author_id": "1111111111",
    "in_reply_to_status_id_str": "1234567888",
    "in_reply_to_screen_name": "third_user",
    "quoted_status_id_str": null
  }
}
```

---

## Summary

This proposal provides a complete twscrape-based solution for bulk scraping 500+ public Twitter profiles:

1. **`bulk_profile_scraper.py`** - Parallel profile scraping with configurable workers
2. **`hydrate_threads.py`** - Recursive parent tweet hydration for thread context
3. **Existing `twitter-processer.py`** - Unchanged, processes the output for LLM consumption

Key advantages:

- Fully automated (no browser required)
- Parallel processing for speed
- Automatic rate limit handling
- Compatible with existing processing pipeline
- Comprehensive thread context through hydration
