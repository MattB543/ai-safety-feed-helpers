#!/usr/bin/env python3
"""Simple check of tweets and URLs in database."""
import os
import sys
import psycopg2
from dotenv import load_dotenv
from collections import Counter
from urllib.parse import urlparse

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

# Use the tweets-specific DB URL if available
db_url = os.getenv('AI_SAFETY_TWEETS_DB_URL') or os.getenv('AI_SAFETY_FEED_DB_URL')
print(f'Using DB: {db_url.split("@")[-1] if db_url else "None"}')

conn = psycopg2.connect(db_url)
cur = conn.cursor()

# Check all schemas
print('=== Schemas ===')
cur.execute("SELECT schema_name FROM information_schema.schemata")
for row in cur.fetchall():
    print(f'  {row[0]}')

# Check for tweets table in all schemas
print('\n=== Looking for tweets tables ===')
cur.execute("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_name LIKE '%tweet%'
""")
for row in cur.fetchall():
    print(f'  {row[0]}.{row[1]}')

# Check twitter_conversation_classifications
print('\n=== Classification table check ===')
try:
    cur.execute("SELECT COUNT(*) FROM twitter_conversation_classifications")
    count = cur.fetchone()[0]
    print(f'  twitter_conversation_classifications: {count} rows')
except Exception as e:
    print(f'  Table not found: {e}')
    conn.rollback()

# Check tweets table directly
print('\n=== Tweets table direct check ===')
try:
    cur.execute("SELECT COUNT(*) FROM tweets")
    count = cur.fetchone()[0]
    print(f'  Total tweets: {count}')

    if count > 0:
        cur.execute("""
            SELECT tweet_id, author_username, text, urls
            FROM tweets
            WHERE created_at >= '2025-11-01'
            LIMIT 5
        """)
        for row in cur.fetchall():
            tid, author, text, urls = row
            text_preview = (text[:80] + '...') if text and len(text) > 80 else text
            print(f'\n  @{author}: {text_preview}')
            print(f'  URLs column: {urls}')
except Exception as e:
    print(f'  Error: {e}')
    conn.rollback()

# If we have tweets, check their URLs
print('\n=== URL Analysis from tweet text ===')
cur.execute("""
    SELECT text, urls
    FROM tweets
    WHERE text LIKE '%http%'
    AND created_at >= '2025-11-01'
    LIMIT 20
""")
rows = cur.fetchall()
print(f'Found {len(rows)} tweets with http in text')

all_urls_from_column = []
for text, urls in rows:
    print(f'\nText: {text[:100]}...' if len(text or '') > 100 else f'Text: {text}')
    print(f'URLs column: {urls}')
    if urls:
        all_urls_from_column.extend(urls if isinstance(urls, list) else [urls])

if all_urls_from_column:
    print(f'\n=== Domain distribution from URLs column ===')
    tco_count = sum(1 for u in all_urls_from_column if 't.co/' in u)
    print(f'Total URLs: {len(all_urls_from_column)}')
    print(f't.co links: {tco_count}')
    print(f'Expanded/real URLs: {len(all_urls_from_column) - tco_count}')

    domains = [urlparse(u).netloc for u in all_urls_from_column]
    print('\nTop domains:')
    for domain, cnt in Counter(domains).most_common(10):
        print(f'  {domain}: {cnt}')

conn.close()
