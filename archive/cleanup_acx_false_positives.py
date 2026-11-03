#!/usr/bin/env python3
"""
cleanup_acx_false_positives.py
──────────────────────────────
Remove recurring non-article Astral Codex Ten formats (open threads, link
roundups, Mantic Monday) that the relevance gate let into `content` on
2026-09-17, and record them in `skipped_posts` so they are not re-ingested.

Dry run by default; pass --apply to delete.
"""
import argparse
import os
import re

import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)

PATTERN = re.compile(
    r"^\s*(open (hidden )?(open )?thread\b|mantic monday\b|links for \w+|model city monday\b|"
    r"highlights from the comments\b|meetups everywhere\b|classifieds thread\b|berkeley meetup\b)",
    re.IGNORECASE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually delete and record skips (default: dry run)")
    ap.add_argument("--source", default="Astral Codex Ten")
    args = ap.parse_args()

    conn = psycopg2.connect(os.environ["AI_SAFETY_FEED_DB_URL"])
    cur = conn.cursor()
    cur.execute("SELECT id, title, title_norm, source_url FROM content WHERE source_type = %s ORDER BY id", (args.source,))
    victims = [r for r in cur.fetchall() if PATTERN.search(r[1] or "")]
    print(f"{len(victims)} rows match the non-article patterns in source '{args.source}':")
    for rid, title, _, url in victims:
        print(f"   {rid}  {title[:70]}")
    if not victims:
        return
    if not args.apply:
        print("\nDry run. Re-run with --apply to delete these rows and record them as skipped.")
        return
    cur.executemany(
        "INSERT INTO skipped_posts (post_id, title_norm, source_url) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        [(f"cleanup:{rid}", title_norm, url) for rid, _, title_norm, url in victims],
    )
    cur.execute("DELETE FROM content WHERE id = ANY(%s)", ([rid for rid, *_ in victims],))
    conn.commit()
    print(f"\nDeleted {len(victims)} rows from content and recorded them in skipped_posts.")
    conn.close()


if __name__ == "__main__":
    main()
