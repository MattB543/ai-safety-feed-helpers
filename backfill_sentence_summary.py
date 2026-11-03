#!/usr/bin/env python3
"""
backfill_sentence_summary.py
──────────────────────────────
Backfill sentence_summary, paragraph_summary, and key_implication for rows that
are missing them, using Gemini (google-genai SDK, model from GEMINI_TEXT_MODEL,
default gemini-3.8-flash).

Exit codes (run_pipeline.py loops on them): 0 = nothing left to try this run, 3 = more
eligible rows remain, 1 = database error or every row in the batch failed. Rows that fail
3 times are skipped from then on (common.RetryState); failures are counted before the API
calls, so a run killed at the stage deadline still counts.
"""

import os
import sys
import time
import logging
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from common import EXIT_DONE, EXIT_FAILED, EXIT_MORE_WORK, RetryState, get_db_connection
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

# Load environment variables
load_dotenv(override=True)

# ───── Runtime settings ───────────────────────────────────────
DB_URL = os.getenv("AI_SAFETY_FEED_DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_TEXT_MODEL") or "gemini-3.8-flash"
BATCH_SIZE = 10  # Process this many rows per run
MAX_INPUT_CHARS = 150000  # Truncate content to avoid API limits

if not (DB_URL and GEMINI_API_KEY):
    sys.exit("Please set AI_SAFETY_FEED_DB_URL and GEMINI_API_KEY in your environment")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)

# ───── Gemini client ─────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 5xx and on 429 rate limits; fail fast on everything else."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    return isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=5, max=60),
    retry=retry_if_exception(_is_retryable),
    before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING),
    reraise=True,
)
def _generate(prompt: str, model_name: str):
    return client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )


def call_gemini_api(prompt: str, model_name: str = GEMINI_MODEL) -> str:
    """Call Gemini with retry on transient errors. Returns text or 'Error: ...'."""
    try:
        response = _generate(prompt, model_name)
        text = (response.text or "").strip()
        if not text:
            logging.error("Gemini returned an empty response")
            return "Error: empty response"
        return text
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return f"Error: {e}"

def summarize_text(text_to_summarize: str) -> str:
    """Generates a concise 1-2 sentence summary using the Gemini API."""
    if not text_to_summarize or text_to_summarize.isspace():
        logging.info("Skipping sentence summary: Input content was empty.")
        return None

    # Truncate overly long inputs
    text_for_model = text_to_summarize[:MAX_INPUT_CHARS]

    prompt = f"""
Summarize the following AI safety content in 2 concise sentences (maximum 50 words).
Focus on the core argument, key insight, or main conclusion rather than methodology.
Use clear, accessible language while preserving technical accuracy.
The summary should be very readable and help readers quickly understand what makes this content valuable or interesting and decide if they want to read more.

--- Content to summarize ---
{text_for_model}
"""
    result = call_gemini_api(prompt)
    if result.startswith("Error:"):
        return None
    return result

def generate_paragraph_summary(text_to_summarize: str) -> str:
    """Generates a structured paragraph summary using the Gemini API."""
    if not text_to_summarize or text_to_summarize.isspace():
        logging.info("Skipping paragraph summary: Input content was empty.")
        return None

    text_for_model = text_to_summarize[:MAX_INPUT_CHARS]

    prompt = f"""
Generate a structured summary of the following AI safety content so the reader can quickly understand the main points. The summary should consist of:

1.  A brief 1-sentence introduction highlighting the main point.
2.  3-5 bullet points covering key arguments, evidence, or insights. Format EACH bullet point as:
    *   **Key concept or term**: Explanation or elaboration of that point.
3.  A brief 1-sentence conclusion with the author's recommendation or final thoughts.

--- Rules ---
-   Make each bullet point concise (1 sentence) and focus on one distinct idea.
-   Bold only the key concept at the start of each bullet, not entire sentences.
-   This format should help readers quickly scan and understand the core content.
-   Only output the summary itself (don't include 'Summary:' or anything else).
-   Use markdown to format the bullet points and improve readability with bolding and italics.
-   Include a double line break after the introduction and before the conclusion.

--- Content to summarize ---
{text_for_model}
"""
    result = call_gemini_api(prompt)
    if result.startswith("Error:"):
        return None
    return result

def generate_key_implication(text_to_analyze: str) -> str:
    """
    Identifies the single most important logical consequence using the Gemini API.
    (Same prompt the scrapers use, so backfilled rows match freshly ingested ones.)
    """
    if not text_to_analyze or text_to_analyze.isspace():
        logging.info("Skipping key implication: Input content was empty.")
        return None

    text_for_model = text_to_analyze[:MAX_INPUT_CHARS]

    prompt = f"""
Based on the AI safety content below, identify the single most important logical consequence or implication in one concise sentence (25-35 words). Focus on:

-   What change in thinking, strategy, or priorities follows from accepting this content's conclusions?
-   How might this alter our understanding of AI safety or governance approaches?
-   A specific actionable insight rather than a general statement of importance.
-   The "so what" that would matter to an informed AI safety community member.

The implication should represent a direct consequence of the content's argument, not simply restate the main point.

--- Content to analyze ---
{text_for_model}
"""
    result = call_gemini_api(prompt)
    if result.startswith("Error:"):
        return None
    return result

# Rows needing work; %s = ids to leave out (retry-skipped, or already failed this run).
ELIGIBLE_SQL = """
    FROM   content
    WHERE  (
               sentence_summary IS NULL
            OR paragraph_summary IS NULL
            OR key_implication IS NULL
           )
           AND (full_content_markdown IS NOT NULL OR full_content IS NOT NULL)
           AND id <> ALL(%s::bigint[])
"""

# ───── Main routine ───────────────────────────────────────────
def main():
    t0 = time.time()
    conn = None
    cur = None
    processed_count = 0
    skipped_count = 0
    retry_state = RetryState("backfill_summary")
    failed_ids, done_ids = [], []
    rows = []
    exit_code = EXIT_DONE

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)

        # Find rows missing any analysis field that have content to summarize.
        logging.info("Fetching rows needing summary/key-implication backfill...")
        cur.execute("""
            SELECT id, title, sentence_summary, paragraph_summary, key_implication,
                   full_content_markdown, full_content
        """ + ELIGIBLE_SQL + """
            ORDER  BY published_date DESC NULLS LAST
            LIMIT  %s
        """, (retry_state.skip_ids(), BATCH_SIZE))

        rows = cur.fetchall()
        if not rows:
            logging.info("✓ All rows have summary/key-implication fields - nothing to backfill.")
            return EXIT_DONE

        logging.info(f"Found {len(rows)} rows to process (model: {GEMINI_MODEL}).")

        # Process each row
        for row in rows:
            row_id = row["id"]
            try:
                retry_state.begin([row_id])  # saved now: still counts if this run gets killed
                logging.info(f"[{row_id}] Processing '{row['title'][:60]}...'")

                # Determine what content to use for analysis
                content = row["full_content_markdown"] or row["full_content"]
                if not content or content.isspace():
                    logging.warning(f"[{row_id}] Skipping - no usable content")
                    skipped_count += 1
                    failed_ids.append(row_id)
                    continue

                # Generate missing summaries
                updates = {}

                if row["sentence_summary"] is None:
                    logging.info(f"   → Generating sentence summary...")
                    sentence_summary = summarize_text(content)
                    if sentence_summary:
                        updates["sentence_summary"] = sentence_summary
                        logging.info(f"   ✓ Sentence summary generated")
                    else:
                        logging.warning(f"   ✗ Failed to generate sentence summary")

                if row["paragraph_summary"] is None:
                    logging.info(f"   → Generating paragraph summary...")
                    paragraph_summary = generate_paragraph_summary(content)
                    if paragraph_summary:
                        updates["paragraph_summary"] = paragraph_summary
                        logging.info(f"   ✓ Paragraph summary generated")
                    else:
                        logging.warning(f"   ✗ Failed to generate paragraph summary")

                if row["key_implication"] is None:
                    logging.info(f"   → Generating key implication...")
                    key_implication = generate_key_implication(content)
                    if key_implication:
                        updates["key_implication"] = key_implication
                        logging.info(f"   ✓ Key implication generated")
                    else:
                        logging.warning(f"   ✗ Failed to generate key implication")

                # Update database if we have any successful generations
                if updates:
                    set_clauses = ", ".join([f"{key}=%s" for key in updates.keys()])
                    values = list(updates.values()) + [row_id]

                    # Summaries changed, so the stored vectors are stale: clear them and run
                    # backfill_embeddings.py next (novelty_analysis needs embedding_full).
                    cur.execute(
                        f"UPDATE content SET {set_clauses}, embedding_short=NULL, embedding_full=NULL WHERE id=%s",
                        values
                    )
                    conn.commit()
                    logging.info(f"   → DB updated successfully for [{row_id}].")
                    processed_count += 1
                else:
                    logging.warning(f"[{row_id}] No updates generated - all API calls failed")
                    skipped_count += 1

                # Done only if every missing field is now filled; a partial update still counts as a failure.
                missing = [k for k in ("sentence_summary", "paragraph_summary", "key_implication") if row[k] is None]
                if all(k in updates for k in missing):
                    done_ids.append(row_id)
                    retry_state.succeeded([row_id])
                else:
                    failed_ids.append(row_id)

                # Small delay to avoid rate limiting
                time.sleep(0.5)

            except Exception as e:
                logging.error(f"Failed to process row [{row_id}]: {e}", exc_info=True)
                if conn:
                    conn.rollback()
                skipped_count += 1
                failed_ids.append(row_id)

        logging.info(f"✓ Finished: {processed_count} updated, {skipped_count} skipped in {time.time() - t0:.1f}s")
        if len(failed_ids) == len(rows):
            logging.error("Every row in the batch failed (%d).", len(rows))
            exit_code = EXIT_FAILED
        else:
            # More work only if eligible rows remain beyond the ones that just failed
            # (those are retried on a later run, up to the RetryState limit).
            cur.execute("SELECT count(*) AS n " + ELIGIBLE_SQL, (retry_state.skip_ids() + failed_ids,))
            remaining = cur.fetchone()["n"]
            logging.info(f"{remaining} eligible row(s) remain.")
            exit_code = EXIT_MORE_WORK if remaining else EXIT_DONE

    except psycopg2.Error as db_err:
        logging.error(f"Database error: {db_err}", exc_info=True)
        if conn:
            conn.rollback()
        exit_code = EXIT_FAILED
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        if conn:
            conn.rollback()
        exit_code = EXIT_FAILED
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logging.info("Database connection closed.")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
