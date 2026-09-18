#!/usr/bin/env python3
"""Verify that the missing URLs are from retweets."""
import os
import sys
import psycopg2
from dotenv import load_dotenv

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()
db_url = os.getenv('AI_SAFETY_TWEETS_DB_URL') or os.getenv('AI_SAFETY_FEED_DB_URL')
conn = psycopg2.connect(db_url)
cur = conn.cursor()

print('=== Analyzing tweets with t.co in text but no stored URLs ===\n')

cur.execute("""
    SELECT interaction_type, COUNT(*) as cnt
    FROM tweets
    WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
      AND text LIKE '%https://t.co/%'
      AND (urls IS NULL OR array_length(urls, 1) IS NULL)
      AND (media_urls IS NULL OR array_length(media_urls, 1) IS NULL)
    GROUP BY interaction_type
    ORDER BY cnt DESC
""")

print('Breakdown by interaction type:')
for itype, cnt in cur.fetchall():
    print(f'  {itype}: {cnt}')

# For retweets, the URLs would be in retweeted_text not extracted separately
print('\n=== Checking if retweet URLs are in retweeted_text ===')
cur.execute("""
    SELECT tweet_id, author_username, text, retweeted_text, retweeted_username
    FROM tweets
    WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
      AND interaction_type = 'retweet'
      AND text LIKE '%https://t.co/%'
      AND (urls IS NULL OR array_length(urls, 1) IS NULL)
    LIMIT 5
""")

for tid, author, text, rt_text, rt_user in cur.fetchall():
    print(f'\n@{author} retweeted @{rt_user}:')
    print(f'  Tweet text: {text[:80]}...' if len(text or '') > 80 else f'  Tweet text: {text}')
    print(f'  RT text: {rt_text[:80]}...' if len(rt_text or '') > 80 else f'  RT text: {rt_text}')

conn.close()
