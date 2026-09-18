#!/usr/bin/env python3
"""
backfill_rt_qt_urls.py
──────────────────────
Backfill URLs for retweets and quote tweets that are missing expanded URLs.

This script:
1. Finds RT/QT tweets with t.co in text but no URLs stored
2. Re-fetches them via twscrape to get the expanded URLs
3. Updates the database with the extracted URLs and media

Usage:
    python backfill_rt_qt_urls.py --dry-run          # Preview what would be updated
    python backfill_rt_qt_urls.py --run              # Actually run the backfill
    python backfill_rt_qt_urls.py --run --limit 100  # Limit to 100 tweets
"""

import asyncio
import argparse
import os
import sys
import logging
import time
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv

try:
    from twscrape import API
except ImportError:
    print("ERROR: twscrape not installed. Run: pip install twscrape")
    sys.exit(1)

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

load_dotenv(override=True)

DATABASE_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL") or os.getenv("AI_SAFETY_FEED_DB_URL")
TWSCRAPE_DB = "accounts.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)


def extract_urls_from_tweet(tweet):
    """Extract all URLs from a tweet, including RT/QT content."""
    urls = []

    # Direct links
    if tweet.links:
        urls.extend([link.url for link in tweet.links if link and link.url])

    # Retweeted content links
    if tweet.retweetedTweet and tweet.retweetedTweet.links:
        urls.extend([link.url for link in tweet.retweetedTweet.links if link and link.url])

    # Quoted content links
    if tweet.quotedTweet and tweet.quotedTweet.links:
        urls.extend([link.url for link in tweet.quotedTweet.links if link and link.url])

    # Deduplicate
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def extract_media_from_tweet(tweet):
    """Extract all media URLs from a tweet, including RT/QT content."""
    media_urls = []

    def extract_from_media(media):
        urls = []
        if media:
            if media.photos:
                urls.extend([p.url for p in media.photos if p and p.url])
            if media.videos:
                urls.extend([v.thumbnailUrl for v in media.videos if v and v.thumbnailUrl])
                # Best quality video
                for v in media.videos:
                    if v.variants:
                        best = max(v.variants, key=lambda x: x.bitrate or 0)
                        if best.url:
                            urls.append(best.url)
            if media.animated:
                for anim in media.animated:
                    if hasattr(anim, 'videoUrl') and anim.videoUrl:
                        urls.append(anim.videoUrl)
                    elif hasattr(anim, 'url') and anim.url:
                        urls.append(anim.url)
        return urls

    # Direct media
    media_urls.extend(extract_from_media(tweet.media))

    # Retweeted media
    if tweet.retweetedTweet:
        media_urls.extend(extract_from_media(tweet.retweetedTweet.media))

    # Quoted media
    if tweet.quotedTweet:
        media_urls.extend(extract_from_media(tweet.quotedTweet.media))

    # Deduplicate
    seen = set()
    return [u for u in media_urls if not (u in seen or seen.add(u))]


def get_tweets_needing_backfill(conn, month: str = None, limit: int = None):
    """Get tweets that have t.co in text but no URLs stored."""
    cur = conn.cursor()

    base_query = """
        SELECT tweet_id, author_username, interaction_type, text
        FROM tweets
        WHERE text LIKE '%%https://t.co/%%'
          AND (urls IS NULL OR array_length(urls, 1) IS NULL)
          AND interaction_type IN ('retweet', 'quote_tweet')
    """

    if month:
        year, mon = month.split('-')
        start_date = f"{year}-{mon}-01"
        next_mon = int(mon) + 1
        next_year = int(year)
        if next_mon > 12:
            next_mon = 1
            next_year += 1
        end_date = f"{next_year}-{next_mon:02d}-01"
        base_query += f" AND created_at >= '{start_date}' AND created_at < '{end_date}'"

    base_query += " ORDER BY created_at DESC"

    if limit:
        base_query += f" LIMIT {limit}"

    cur.execute(base_query)
    return cur.fetchall()


def update_tweet_urls(conn, tweet_id: str, urls: list, media_urls: list):
    """Update a tweet's URLs and media_urls in the database."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE tweets
        SET urls = %s, media_urls = %s
        WHERE tweet_id = %s
    """, (urls if urls else None, media_urls if media_urls else None, tweet_id))
    conn.commit()
    return cur.rowcount


async def run_backfill(month: str = None, limit: int = None, dry_run: bool = True):
    """Main backfill execution."""

    print("\n" + "=" * 60)
    print("BACKFILL RT/QT URLs")
    print("=" * 60)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (updating database)'}")
    if month:
        print(f"Month filter: {month}")
    if limit:
        print(f"Limit: {limit}")
    print("=" * 60)

    # Initialize twscrape API
    api = API(TWSCRAPE_DB)

    # Check account pool
    try:
        pool_stats = await api.pool.stats()
        logger.info(f"Account Pool: {pool_stats}")
        if pool_stats.get('active', 0) == 0:
            logger.error("No active Twitter accounts!")
            return
    except Exception as e:
        logger.error(f"Failed to check account pool: {e}")
        return

    # Connect to database
    conn = psycopg2.connect(DATABASE_URL)

    # Get tweets needing backfill
    tweets = get_tweets_needing_backfill(conn, month, limit)
    total = len(tweets)

    print(f"\nFound {total} tweets needing URL backfill\n")

    if total == 0:
        print("Nothing to backfill!")
        conn.close()
        return

    # Process tweets
    stats = {"updated": 0, "no_urls": 0, "errors": 0, "urls_found": 0}
    start_time = time.time()

    for i, (tweet_id, author, itype, text) in enumerate(tweets):
        try:
            # Fetch tweet details
            tweet = await api.tweet_details(int(tweet_id))

            if not tweet:
                logger.warning(f"[{i+1}/{total}] Tweet {tweet_id} not found")
                stats["errors"] += 1
                continue

            # Extract URLs
            urls = extract_urls_from_tweet(tweet)
            media_urls = extract_media_from_tweet(tweet)

            stats["urls_found"] += len(urls)

            # Show what we found
            text_preview = text[:50] + "..." if len(text) > 50 else text

            if urls or media_urls:
                print(f"[{i+1}/{total}] @{author} ({itype})")
                print(f"  Text: {text_preview}")
                if urls:
                    print(f"  URLs found: {urls}")
                if media_urls:
                    print(f"  Media found: {len(media_urls)} items")

                if not dry_run:
                    update_tweet_urls(conn, tweet_id, urls, media_urls)
                    stats["updated"] += 1
                else:
                    stats["updated"] += 1  # Would be updated
            else:
                stats["no_urls"] += 1

            # Rate limiting - be gentle
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                print(f"  ... processed {i+1}/{total} ({rate:.1f}/min)")
                await asyncio.sleep(0.5)

        except Exception as e:
            logger.warning(f"[{i+1}/{total}] Error processing {tweet_id}: {e}")
            stats["errors"] += 1

    conn.close()

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Total tweets processed: {total}")
    print(f"  {'Would update' if dry_run else 'Updated'}: {stats['updated']}")
    print(f"  No URLs found: {stats['no_urls']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Total URLs extracted: {stats['urls_found']}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    if dry_run:
        print("\n  ** DRY RUN - no changes made. Use --run to apply. **")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill URLs for RT/QT tweets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without updating database"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually update the database"
    )
    parser.add_argument(
        "--month",
        type=str,
        default="2025-11",
        help="Month to process (YYYY-MM format, default: 2025-11)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of tweets to process"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.run:
        parser.print_help()
        print("\nError: Please specify --dry-run or --run")
        sys.exit(1)

    dry_run = not args.run

    asyncio.run(run_backfill(
        month=args.month,
        limit=args.limit,
        dry_run=dry_run
    ))


if __name__ == "__main__":
    main()
