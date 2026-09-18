#!/usr/bin/env python3
"""Comprehensive URL analysis of November tweets."""
import os
import sys
import re
import psycopg2
from dotenv import load_dotenv
from collections import Counter
from urllib.parse import urlparse

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()
db_url = os.getenv('AI_SAFETY_TWEETS_DB_URL') or os.getenv('AI_SAFETY_FEED_DB_URL')
conn = psycopg2.connect(db_url)
cur = conn.cursor()

print('=== November 2025 URL Analysis ===\n')

# Count tweets by URL status
cur.execute("""
    SELECT
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE urls IS NOT NULL AND array_length(urls, 1) > 0) as has_urls_column,
        COUNT(*) FILTER (WHERE text LIKE '%https://t.co/%') as has_tco_in_text,
        COUNT(*) FILTER (WHERE text LIKE '%http%' AND text NOT LIKE '%https://t.co/%') as has_other_url_in_text
    FROM tweets
    WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
""")
row = cur.fetchone()
total, has_urls_col, has_tco, has_other = row

print(f'Total November tweets: {total:,}')
print(f'  With URLs in column: {has_urls_col:,} ({100*has_urls_col/total:.1f}%)')
print(f'  With t.co in text: {has_tco:,} ({100*has_tco/total:.1f}%)')
print(f'  With other http in text: {has_other:,} ({100*has_other/total:.1f}%)')

# Get all URLs from the urls column
print('\n=== Extracting all URLs from column ===')
cur.execute("""
    SELECT urls
    FROM tweets
    WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
      AND urls IS NOT NULL AND array_length(urls, 1) > 0
""")

all_urls = []
for row in cur.fetchall():
    if row[0]:
        all_urls.extend(row[0])

print(f'Total URLs extracted: {len(all_urls):,}')

# Check for t.co links in the urls column
tco_count = sum(1 for u in all_urls if 't.co/' in u)
print(f't.co links in column: {tco_count}')
print(f'Expanded URLs: {len(all_urls) - tco_count}')

# Domain analysis
print('\n=== Top Domains ===')
domains = []
for url in all_urls:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        domains.append(domain)
    except:
        pass

for domain, count in Counter(domains).most_common(30):
    print(f'  {count:4d}  {domain}')

# Top specific URLs
print('\n=== Top Individual URLs ===')
url_counts = Counter(all_urls)
for url, count in url_counts.most_common(20):
    print(f'  {count:4d}  {url[:80]}...' if len(url) > 80 else f'  {count:4d}  {url}')

# Check what types of content have URLs
print('\n=== URLs by Interaction Type ===')
cur.execute("""
    SELECT
        interaction_type,
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE urls IS NOT NULL AND array_length(urls, 1) > 0) as with_urls
    FROM tweets
    WHERE created_at >= '2025-11-01' AND created_at < '2025-12-01'
    GROUP BY interaction_type
    ORDER BY total DESC
""")
for row in cur.fetchall():
    itype, total_count, with_urls = row
    pct = 100*with_urls/total_count if total_count > 0 else 0
    print(f'  {itype:12s}: {with_urls:,}/{total_count:,} ({pct:.1f}%)')

conn.close()
print('\n=== Done ===')
