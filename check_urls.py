#!/usr/bin/env python3
"""Quick script to check what URLs are stored in the database."""
import os
import sys
import psycopg2
from dotenv import load_dotenv
from collections import Counter
from urllib.parse import urlparse

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

# Show the DB URL (masked)
db_url = os.getenv('AI_SAFETY_FEED_DB_URL')
if db_url:
    # Mask password for display
    parts = db_url.split('@')
    if len(parts) > 1:
        print(f'DB URL: ***@{parts[-1]}')
    else:
        print(f'DB URL: {db_url[:50]}...')

conn = psycopg2.connect(db_url)
cur = conn.cursor()

# Check all tables in DB
print('=== All Tables in Database ===')
cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
""")
tables = [row[0] for row in cur.fetchall()]
for t in tables:
    cur.execute(f'SELECT COUNT(*) FROM "{t}"')
    count = cur.fetchone()[0]
    print(f'  {t}: {count:,} rows')

print('\n=== Table Schema (tweets) ===')
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'tweets'
    ORDER BY ordinal_position
""")
for col, dtype in cur.fetchall():
    print(f'  {col}: {dtype}')

# Also check fellowship_mvp schema
print('\n=== Table Schema (fellowship_mvp) ===')
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'fellowship_mvp'
    ORDER BY ordinal_position
""")
for col, dtype in cur.fetchall():
    print(f'  {col}: {dtype}')

# Check sample from fellowship_mvp
print('\n=== Sample from fellowship_mvp ===')
cur.execute("""
    SELECT * FROM fellowship_mvp LIMIT 3
""")
cols = [desc[0] for desc in cur.description]
print(f'Columns: {cols}')
for row in cur.fetchall():
    print(f'\n{dict(zip(cols, row))}')

# Check total tweets and how many have urls
print('\n=== URL Column Stats ===')
cur.execute("SELECT COUNT(*) FROM tweets")
total = cur.fetchone()[0]
print(f'Total tweets: {total}')

cur.execute("SELECT COUNT(*) FROM tweets WHERE urls IS NOT NULL")
with_urls = cur.fetchone()[0]
print(f'Tweets with urls NOT NULL: {with_urls}')

cur.execute("SELECT COUNT(*) FROM tweets WHERE urls IS NOT NULL AND urls != '{}'")
with_nonempty_urls = cur.fetchone()[0]
print(f'Tweets with non-empty urls array: {with_nonempty_urls}')

# Sample some raw URL data
print('\n=== Sample Raw URL Data ===')
cur.execute('''
    SELECT tweet_id, author_username, urls, text
    FROM tweets
    WHERE urls IS NOT NULL AND urls != '{}'
    LIMIT 10
''')

all_urls = []
for row in cur.fetchall():
    tweet_id, author, urls, text = row
    print(f'\n@{author} (tweet {tweet_id}):')
    print(f'  Text: {text[:100]}...' if text and len(text) > 100 else f'  Text: {text}')
    print(f'  URLs: {urls}')
    if urls:
        for url in (urls if isinstance(urls, list) else [urls])[:3]:
            all_urls.append(url)

if not all_urls:
    # Maybe try looking at tweets with URLs in text
    print('\n=== Checking tweets with http in text ===')
    cur.execute('''
        SELECT tweet_id, author_username, urls, text
        FROM tweets
        WHERE text LIKE '%http%'
          AND created_at >= '2025-11-01'
        LIMIT 10
    ''')
    for row in cur.fetchall():
        tweet_id, author, urls, text = row
        print(f'\n@{author}:')
        print(f'  Text: {text[:150]}...' if text and len(text) > 150 else f'  Text: {text}')
        print(f'  URLs column: {urls}')

# Check if any are t.co links
if all_urls:
    print('\n\n=== URL Analysis ===')
    tco_count = sum(1 for u in all_urls if 't.co/' in u)
    real_count = len(all_urls) - tco_count
    print(f'Total URLs: {len(all_urls)}')
    print(f't.co links: {tco_count}')
    print(f'Real/expanded URLs: {real_count}')

    # Get domain distribution
    print('\n=== Domain Distribution ===')
    domains = [urlparse(u).netloc for u in all_urls]
    for domain, count in Counter(domains).most_common(15):
        print(f'  {domain}: {count}')

conn.close()
