#!/usr/bin/env python3
"""
backfill_embeddings.py
──────────────────────
Fill embedding_short / embedding_full for content rows that lack them, using the
same inputs the scrapers use:
  • short vector: the title (cleaned_title when present)
  • full vector:  sentence summary + paragraph summary + key implication + topics
                  (see llm_common.build_embedding_text)

Rows whose summaries are all still NULL are skipped; run
backfill_sentence_summary.py first so the full vector has something to embed.
"""

import logging
import os
import sys

import psycopg2
from psycopg2 import extras
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv(override=True)

from llm_common import build_embedding_text, generate_embeddings_batch, require_env  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
require_env(azure=False, embeddings=True)

DB_URL = os.environ.get("AI_SAFETY_FEED_DB_URL")
if not DB_URL:
    sys.exit("Set AI_SAFETY_FEED_DB_URL")

BATCH = 100  # rows per embeddings request (2 vectors per row)


def main() -> None:
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    updated = skipped = failed = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, cleaned_title, sentence_summary, paragraph_summary,
                       key_implication, topics
                FROM   content
                WHERE  embedding_short IS NULL OR embedding_full IS NULL
                ORDER  BY id
                """
            )
            rows = cur.fetchall()
        logging.info("%d rows need embeddings", len(rows))

        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            ids, shorts, fulls = [], [], []
            for rid, title, cleaned_title, sentence, paragraph, implication, topics in chunk:
                short_text = (cleaned_title or title or "").strip()
                if not ((sentence or "").strip() or (paragraph or "").strip() or (implication or "").strip()):
                    logging.warning("[%s] skipped: no summaries yet (run backfill_sentence_summary.py first)", rid)
                    skipped += 1
                    continue
                full_text = build_embedding_text(sentence, paragraph, implication, topics or [])
                if not short_text:
                    logging.warning("[%s] skipped: no title", rid)
                    skipped += 1
                    continue
                ids.append(rid)
                shorts.append(short_text)
                fulls.append(full_text)
            if not ids:
                continue

            try:
                vectors = generate_embeddings_batch(shorts, fulls)
            except Exception as e:
                logging.error("Embedding batch failed (ids %s..%s): %s", ids[0], ids[-1], e)
                failed += len(ids)
                continue

            updates = [
                (emb_short, emb_full, rid)
                for rid, (emb_short, emb_full) in zip(ids, vectors)
                if emb_short is not None and emb_full is not None
            ]
            with conn.cursor() as cur:
                extras.execute_batch(
                    cur,
                    "UPDATE content SET embedding_short=%s, embedding_full=%s WHERE id=%s",
                    updates,
                )
            conn.commit()
            updated += len(updates)
            logging.info("Updated %d rows (running total %d)", len(updates), updated)
    except Exception as e:
        logging.error("Backfill aborted: %s", e, exc_info=True)
        conn.rollback()
    finally:
        conn.close()
    logging.info("Done: %d updated, %d skipped, %d failed", updated, skipped, failed)


if __name__ == "__main__":
    main()
