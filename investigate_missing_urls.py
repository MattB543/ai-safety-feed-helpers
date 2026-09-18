#!/usr/bin/env python3
"""
Investigate tweets with t.co in text but no URLs in column.
Re-scrape a few to see what data twscrape provides.
"""
import os
import sys
import asyncio
import json
import psycopg2
from dotenv import load_dotenv

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

# Database connection
db_url = os.getenv('AI_SAFETY_TWEETS_DB_URL') or os.getenv('AI_SAFETY_FEED_DB_URL')
conn = psycopg2.connect(db_url)
cur = conn.cursor()

print('=== Finding tweets with t.co in text but NULL urls column ===\n')

# Get sample tweets with t.co in text but no URLs stored
cur.execute("""
    SELECT tweet_id, author_username, text, urls, media_urls
    FROM tweets
    WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
      AND text LIKE '%https://t.co/%'
      AND (urls IS NULL OR array_length(urls, 1) IS NULL)
    LIMIT 20
""")

missing_url_tweets = cur.fetchall()
print(f'Found {len(missing_url_tweets)} sample tweets with t.co but no urls\n')

for i, (tid, author, text, urls, media_urls) in enumerate(missing_url_tweets[:10]):
    print(f'\n--- Tweet {i+1}: {tid} by @{author} ---')
    print(f'Text: {text[:200]}...' if len(text or '') > 200 else f'Text: {text}')
    print(f'URLs column: {urls}')
    print(f'Media URLs column: {media_urls}')

# Now let's re-scrape these tweets to see what twscrape gives us
print('\n\n' + '='*60)
print('=== Re-scraping sample tweets to check available data ===')
print('='*60)

# Get tweet IDs to re-scrape
tweet_ids_to_check = [row[0] for row in missing_url_tweets[:5]]

async def check_tweet_data():
    from twscrape import API

    api = API("accounts.db")

    for tweet_id in tweet_ids_to_check:
        print(f'\n\n--- Re-scraping tweet {tweet_id} ---')
        try:
            tweet = await api.tweet_details(int(tweet_id))
            if tweet:
                print(f'Author: @{tweet.user.username if tweet.user else "unknown"}')
                print(f'Text: {tweet.rawContent[:150]}...' if len(tweet.rawContent or '') > 150 else f'Text: {tweet.rawContent}')

                # Check links
                print(f'\ntweet.links: {tweet.links}')
                if tweet.links:
                    for link in tweet.links:
                        print(f'  - url (expanded): {link.url}')
                        print(f'    text (display): {link.text}')
                        print(f'    tcourl: {link.tcourl}')

                # Check media
                print(f'\ntweet.media: {tweet.media}')
                if tweet.media:
                    if tweet.media.photos:
                        print(f'  Photos: {len(tweet.media.photos)}')
                        for photo in tweet.media.photos:
                            print(f'    - {photo.url}')
                    if tweet.media.videos:
                        print(f'  Videos: {len(tweet.media.videos)}')
                        for video in tweet.media.videos:
                            print(f'    - thumbnail: {video.thumbnailUrl}')
                            print(f'    - variants: {video.variants}')
                    if tweet.media.animated:
                        print(f'  Animated (GIFs): {len(tweet.media.animated)}')
                        for anim in tweet.media.animated:
                            print(f'    - {anim}')

                # Check for card/preview data
                print(f'\ntweet.card: {tweet.card}')

                # Raw data exploration
                print(f'\n--- All tweet attributes ---')
                for attr in dir(tweet):
                    if not attr.startswith('_'):
                        try:
                            val = getattr(tweet, attr)
                            if val is not None and not callable(val):
                                val_str = str(val)
                                if len(val_str) > 100:
                                    val_str = val_str[:100] + '...'
                                print(f'  {attr}: {val_str}')
                        except:
                            pass
            else:
                print(f'  Tweet not found or unavailable')
        except Exception as e:
            print(f'  Error: {e}')

asyncio.run(check_tweet_data())

conn.close()
