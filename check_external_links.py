#!/usr/bin/env python3
"""
Check tweets with real external URLs to verify we're capturing them properly.
Also check what additional media data we could be storing.
"""
import os
import sys
import asyncio
import psycopg2
from dotenv import load_dotenv

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

db_url = os.getenv('AI_SAFETY_TWEETS_DB_URL') or os.getenv('AI_SAFETY_FEED_DB_URL')
conn = psycopg2.connect(db_url)
cur = conn.cursor()

print('=== Checking tweets WITH URLs in column ===\n')

# Get tweets that DO have URLs stored
cur.execute("""
    SELECT tweet_id, author_username, text, urls, media_urls
    FROM tweets
    WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
      AND urls IS NOT NULL AND array_length(urls, 1) > 0
    LIMIT 10
""")

for tid, author, text, urls, media_urls in cur.fetchall():
    print(f'\n--- Tweet {tid} by @{author} ---')
    print(f'Text: {text[:150]}...' if len(text or '') > 150 else f'Text: {text}')
    print(f'URLs: {urls}')
    print(f'Media URLs: {media_urls}')

# Re-scrape one to verify
print('\n\n' + '='*60)
print('=== Re-scraping a tweet with external URLs ===')
print('='*60)

# Get a tweet ID with URLs
cur.execute("""
    SELECT tweet_id FROM tweets
    WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
      AND urls IS NOT NULL AND array_length(urls, 1) > 0
    LIMIT 1
""")
tweet_id = cur.fetchone()[0]

async def check_tweet():
    from twscrape import API
    api = API("accounts.db")

    tweet = await api.tweet_details(int(tweet_id))
    if tweet:
        print(f'\nTweet ID: {tweet_id}')
        print(f'Author: @{tweet.user.username}')
        print(f'Text: {tweet.rawContent[:200]}...' if len(tweet.rawContent or '') > 200 else f'Text: {tweet.rawContent}')

        print(f'\nLinks (external URLs):')
        if tweet.links:
            for link in tweet.links:
                print(f'  expanded_url: {link.url}')
                print(f'  display_url:  {link.text}')
                print(f'  t.co_url:     {link.tcourl}')
        else:
            print('  (none)')

        print(f'\nMedia:')
        if tweet.media:
            if tweet.media.photos:
                print(f'  Photos ({len(tweet.media.photos)}):')
                for p in tweet.media.photos:
                    print(f'    - {p.url}')
            if tweet.media.videos:
                print(f'  Videos ({len(tweet.media.videos)}):')
                for v in tweet.media.videos:
                    print(f'    - thumbnail: {v.thumbnailUrl}')
                    if v.variants:
                        best = max(v.variants, key=lambda x: x.bitrate or 0)
                        print(f'    - best quality: {best.url}')
            if tweet.media.animated:
                print(f'  GIFs ({len(tweet.media.animated)}):')
                for a in tweet.media.animated:
                    print(f'    - {a}')
        else:
            print('  (none)')

        print(f'\nCard (URL preview):')
        if tweet.card:
            print(f'  {tweet.card}')
        else:
            print('  (none)')

asyncio.run(check_tweet())

# Summary statistics
print('\n\n' + '='*60)
print('=== Summary: What data is available ===')
print('='*60)

cur.execute("""
    SELECT
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE urls IS NOT NULL AND array_length(urls, 1) > 0) as has_ext_urls,
        COUNT(*) FILTER (WHERE media_urls IS NOT NULL AND array_length(media_urls, 1) > 0) as has_media_urls,
        COUNT(*) FILTER (WHERE text LIKE '%https://t.co/%') as has_tco_in_text
    FROM tweets
    WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
""")
total, has_ext, has_media, has_tco = cur.fetchone()

print(f'\nNovember 2025 tweets: {total:,}')
print(f'  With external URLs stored: {has_ext:,} ({100*has_ext/total:.1f}%)')
print(f'  With media URLs stored: {has_media:,} ({100*has_media/total:.1f}%)')
print(f'  With any t.co in text: {has_tco:,} ({100*has_tco/total:.1f}%)')

# Check overlap
cur.execute("""
    SELECT COUNT(*) FROM tweets
    WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
      AND text LIKE '%https://t.co/%'
      AND (urls IS NULL OR array_length(urls, 1) IS NULL)
      AND (media_urls IS NULL OR array_length(media_urls, 1) IS NULL)
""")
missing_both = cur.fetchone()[0]
print(f'\n  t.co in text but NEITHER urls nor media_urls stored: {missing_both:,}')

# Check what those are
if missing_both > 0:
    print(f'\n  Sampling tweets with t.co but no stored URLs or media:')
    cur.execute("""
        SELECT tweet_id, author_username, text
        FROM tweets
        WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
          AND text LIKE '%https://t.co/%'
          AND (urls IS NULL OR array_length(urls, 1) IS NULL)
          AND (media_urls IS NULL OR array_length(media_urls, 1) IS NULL)
        LIMIT 5
    """)
    for tid, author, text in cur.fetchall():
        print(f'\n    @{author}: {text[:100]}...' if len(text or '') > 100 else f'\n    @{author}: {text}')

conn.close()
