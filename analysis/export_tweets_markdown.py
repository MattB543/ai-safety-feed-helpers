#!/usr/bin/env python3
"""
export_tweets_markdown.py
─────────────────────────
Export tweets from database to a clean chronological markdown file.

Usage:
    python export_tweets_markdown.py
    python export_tweets_markdown.py --output tweets.md
    python export_tweets_markdown.py --profile jackclarksf
"""

import argparse
import os
import re
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)
DATABASE_URL = os.getenv("AI_SAFETY_FEED_DB_URL")


def replace_tco_links(text: str, urls: list) -> str:
    """Replace t.co links in text with expanded URLs from the urls list."""
    if not text or not urls:
        return text

    # Find all t.co links in the text
    tco_pattern = r'https://t\.co/\w+'
    tco_links = re.findall(tco_pattern, text)

    if not tco_links:
        return text

    # Replace each t.co link with expanded URL (in order)
    result = text
    for i, tco_link in enumerate(tco_links):
        if i < len(urls):
            # Replace with markdown link
            expanded_url = urls[i]
            result = result.replace(tco_link, expanded_url, 1)

    return result


def format_tweet(row, plain: bool = False) -> str:
    """Format a single tweet as markdown."""
    (tweet_id, author, author_display, text, created_at, interaction_type,
     in_reply_to_user, quoted_text, quoted_user, retweeted_text, retweeted_user,
     favorite_count, retweet_count, source_profile, is_hydrated, urls) = row

    # Replace t.co links with expanded URLs
    text = replace_tco_links(text, urls)
    quoted_text = replace_tco_links(quoted_text, urls)
    retweeted_text = replace_tco_links(retweeted_text, urls)

    # Build the tweet block
    lines = []

    if plain:
        # Plain format: just @author and text, no formatting
        lines.append(f"@{author}")
        if interaction_type == "retweet" and retweeted_text:
            lines.append(f"RT @{retweeted_user or 'unknown'}: {retweeted_text}")
        elif interaction_type == "quote_tweet" and quoted_text:
            if text:
                lines.append(text)
            lines.append(f"QT @{quoted_user or 'unknown'}: {quoted_text}")
        elif interaction_type == "reply":
            if in_reply_to_user:
                lines.append(f"Reply to @{in_reply_to_user}")
            lines.append(text or "")
        else:
            lines.append(text or "")
        lines.append("")
        lines.append("---")
        lines.append("")
    else:
        # Full markdown format
        date_str = created_at.strftime("%Y-%m-%d %H:%M") if created_at else "Unknown date"
        display = author_display or author
        lines.append(f"### @{author} ({display})")
        lines.append(f"*{date_str}*")
        lines.append("")

        if interaction_type == "retweet" and retweeted_text:
            lines.append(f"**Retweeted @{retweeted_user or 'unknown'}:**")
            lines.append(f"> {retweeted_text}")
        elif interaction_type == "quote_tweet" and quoted_text:
            if text:
                lines.append(text)
                lines.append("")
            lines.append(f"**Quoting @{quoted_user or 'unknown'}:**")
            lines.append(f"> {quoted_text}")
        elif interaction_type == "reply":
            if in_reply_to_user:
                lines.append(f"*Replying to @{in_reply_to_user}*")
                lines.append("")
            lines.append(text or "")
        else:
            lines.append(text or "")

        if favorite_count or retweet_count:
            lines.append("")
            stats = []
            if favorite_count:
                stats.append(f"{favorite_count:,} likes")
            if retweet_count:
                stats.append(f"{retweet_count:,} retweets")
            lines.append(f"*{' | '.join(stats)}*")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def export_to_markdown(output_file: str, profile: str = None, include_hydrated: bool = False, plain: bool = False, month: str = None, ai_safety_only: bool = False):
    """Export tweets to markdown file."""

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Build query
    if ai_safety_only:
        # Join with classifications table to filter by AI safety
        query = """
            SELECT
                t.tweet_id, t.author_username, t.author_display_name, t.text, t.created_at,
                t.interaction_type, t.in_reply_to_username,
                t.quoted_text, t.quoted_username,
                t.retweeted_text, t.retweeted_username,
                t.favorite_count, t.retweet_count,
                t.source_profile, t.is_hydrated_parent,
                t.urls
            FROM tweets t
            INNER JOIN twitter_conversation_classifications c
                ON t.conversation_id = c.conversation_id
            WHERE c.is_ai_safety = TRUE
        """
    else:
        query = """
            SELECT
                tweet_id, author_username, author_display_name, text, created_at,
                interaction_type, in_reply_to_username,
                quoted_text, quoted_username,
                retweeted_text, retweeted_username,
                favorite_count, retweet_count,
                source_profile, is_hydrated_parent,
                urls
            FROM tweets
            WHERE 1=1
        """
    params = []

    if not include_hydrated:
        if ai_safety_only:
            query += " AND t.is_hydrated_parent = FALSE"
        else:
            query += " AND is_hydrated_parent = FALSE"

    if profile:
        if ai_safety_only:
            query += " AND t.source_profile = %s"
        else:
            query += " AND source_profile = %s"
        params.append(profile)

    if month:
        year, mon = month.split('-')
        start_date = f"{year}-{mon}-01"
        next_mon = int(mon) + 1
        next_year = int(year)
        if next_mon > 12:
            next_mon = 1
            next_year += 1
        end_date = f"{next_year}-{next_mon:02d}-01"
        if ai_safety_only:
            query += " AND t.created_at >= %s AND t.created_at < %s"
        else:
            query += " AND created_at >= %s AND created_at < %s"
        params.extend([start_date, end_date])

    if ai_safety_only:
        query += " ORDER BY t.created_at ASC"
    else:
        query += " ORDER BY created_at ASC"

    cur.execute(query, params)
    rows = cur.fetchall()

    # Build content
    lines = []

    if not plain:
        lines.append("# Twitter Feed Export")
        lines.append("")
        lines.append(f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        if profile:
            lines.append(f"*Profile: @{profile}*")
        if month:
            lines.append(f"*Month: {month}*")
        lines.append(f"*Total tweets: {len(rows)}*")
        lines.append("")
        lines.append("---")
        lines.append("")

    for row in rows:
        lines.append(format_tweet(row, plain=plain))

    # Write to file
    content = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Exported {len(rows)} tweets to {output_file}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Export tweets to markdown")
    parser.add_argument(
        "--output", "-o",
        default="tweets_export.md",
        help="Output markdown file (default: tweets_export.md)"
    )
    parser.add_argument(
        "--profile", "-p",
        help="Filter by source profile (e.g., jackclarksf)"
    )
    parser.add_argument(
        "--include-hydrated",
        action="store_true",
        help="Include hydrated parent tweets"
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Plain text format: no markdown formatting, timestamps, or stats"
    )
    parser.add_argument(
        "--month",
        type=str,
        help="Filter by month (YYYY-MM format, e.g., 2025-11)"
    )
    parser.add_argument(
        "--ai-safety",
        action="store_true",
        help="Only include tweets classified as AI safety related"
    )

    args = parser.parse_args()
    export_to_markdown(args.output, args.profile, args.include_hydrated, args.plain, args.month, args.ai_safety)


if __name__ == "__main__":
    main()
