#!/usr/bin/env python3
"""
export_url_stats.py
───────────────────
Export URL statistics from tweets to CSV files.

Outputs:
  - domain_counts.csv: Count of each domain (normalized)
  - url_counts.csv: Count of each specific URL

Usage:
    python export_url_stats.py
    python export_url_stats.py --month 2025-11
    python export_url_stats.py --output-dir ./stats
"""

import argparse
import csv
import os
import re
from collections import Counter
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)
DATABASE_URL = os.getenv("AI_SAFETY_FEED_DB_URL") or os.getenv("AI_SAFETY_TWEETS_DB_URL")


def normalize_domain(domain: str) -> str:
    """Normalize domain by removing www. prefix and lowercasing."""
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def normalize_url(url: str) -> str:
    """Normalize URL by removing tracking params and standardizing."""
    if not url:
        return url

    # Remove common tracking parameters
    url = re.sub(r'[?&](utm_\w+|ref|source|fbclid|gclid|mc_\w+)=[^&]*', '', url)
    # Clean up any leftover ? or & at the end
    url = re.sub(r'[?&]$', '', url)

    return url


def extract_domain(url: str) -> str:
    """Extract and normalize domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        return normalize_domain(domain)
    except Exception:
        return None


def get_urls_from_db(month: str = None):
    """Fetch all URLs from the database."""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    query = """
        SELECT urls
        FROM tweets
        WHERE urls IS NOT NULL AND array_length(urls, 1) > 0
    """
    params = []

    if month:
        year, mon = month.split('-')
        start_date = f"{year}-{mon}-01"
        next_mon = int(mon) + 1
        next_year = int(year)
        if next_mon > 12:
            next_mon = 1
            next_year += 1
        end_date = f"{next_year}-{next_mon:02d}-01"
        query += " AND created_at >= %s AND created_at < %s"
        params.extend([start_date, end_date])

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    # Flatten all URLs
    all_urls = []
    for (urls,) in rows:
        if urls:
            all_urls.extend(urls)

    return all_urls


def export_stats(month: str = None, output_dir: str = "."):
    """Export domain and URL statistics to CSV files."""

    print(f"Fetching URLs from database...")
    if month:
        print(f"  Month filter: {month}")

    all_urls = get_urls_from_db(month)
    print(f"  Found {len(all_urls):,} total URLs")

    # Count domains
    domain_counter = Counter()
    url_counter = Counter()

    for url in all_urls:
        # Normalize and count URL
        normalized_url = normalize_url(url)
        url_counter[normalized_url] += 1

        # Extract and count domain
        domain = extract_domain(url)
        if domain:
            domain_counter[domain] += 1

    # Filter out x.com/twitter.com self-references (optional - keep for now)
    # We'll keep all domains for completeness

    print(f"  Unique domains: {len(domain_counter):,}")
    print(f"  Unique URLs: {len(url_counter):,}")

    # Prepare output filenames
    suffix = f"_{month}" if month else ""
    domain_file = os.path.join(output_dir, f"domain_counts{suffix}.csv")
    url_file = os.path.join(output_dir, f"url_counts{suffix}.csv")

    # Export domain counts
    print(f"\nWriting {domain_file}...")
    with open(domain_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'count'])
        for domain, count in domain_counter.most_common():
            writer.writerow([domain, count])

    # Export URL counts
    print(f"Writing {url_file}...")
    with open(url_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'count'])
        for url, count in url_counter.most_common():
            writer.writerow([url, count])

    # Print top domains
    print(f"\nTop 20 domains:")
    for domain, count in domain_counter.most_common(20):
        print(f"  {count:5,}  {domain}")

    print(f"\nDone! Files written:")
    print(f"  - {domain_file}")
    print(f"  - {url_file}")


def main():
    parser = argparse.ArgumentParser(description="Export URL statistics to CSV")
    parser.add_argument(
        "--month",
        type=str,
        help="Filter by month (YYYY-MM format, e.g., 2025-11)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for CSV files (default: current directory)"
    )

    args = parser.parse_args()

    # Create output directory if needed
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    export_stats(month=args.month, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
