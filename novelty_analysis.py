#!/usr/bin/env python3
"""
backfill_novelty_openai.py
──────────────────────────
Compute a "uniqueness / novelty" score plus a short
'what-this-adds' note for the newest posts that still lack them,
using **OpenAI GPT‑4o** instead of Google Gemini.
"""

# ───── Imports ────────────────────────────────────────────────
import os, sys, time, json, logging, psycopg2
from psycopg2 import extras
from openai import OpenAI  # OpenAI Python v1.x client
from datetime import datetime

# ───── Runtime settings ───────────────────────────────────────
DB_URL              = os.getenv("AI_SAFETY_FEED_DB_URL")
OPENAI_API_KEY      = os.getenv("OPEN_AI_FREE_CREDITS_KEY")
MODEL               = "gpt-4.1"   # pick the tier that matches your quota
BATCH               = 7              # rows to back‑fill each run
K_NEIGHBOURS        = 20              # vector recall set size

if not (DB_URL and OPENAI_API_KEY):
    sys.exit("Set AI_SAFETY_FEED_DB_URL and OPEN_AI_FREE_CREDITS_KEY")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s"
)

# ───── OpenAI client ─────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)


def openai_json(prompt: str,
                *,
                model: str | None = None,
                temperature: float = 0.15,
                max_tokens: int = 2048):
    """Return parsed JSON (or None) using GPT‑4o in strict‑JSON mode."""
    sys_prompt = (
        "You are a service that MUST return **only** valid minified JSON. "
        "Do not wrap the JSON in markdown or add any commentary."
    )
    try:
        target_model = model or MODEL
        logging.info("Using model: %s", target_model)

        rsp = client.chat.completions.create(
            model           = target_model,
            messages        = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": prompt}
            ],
            temperature     = temperature,
            max_tokens      = max_tokens,
            response_format = {"type": "json_object"}
        )
        raw_text = rsp.choices[0].message.content
        logging.info("Raw GPT response text: %s", raw_text)

        return json.loads(raw_text)

    except json.JSONDecodeError as json_err:
        logging.warning(f"OpenAI JSON failure – Parsing error: {json_err}. Text was: {raw_text[:200]}…")
        return None
    except Exception as e:
        logging.warning("OpenAI JSON failure – %s", e)
        return None

# ───── Prompt templates (unchanged from original) ────────────
OVERLAP_PROMPT = """You are an expert AI‑safety editor.

REFERENCE POST - MAIN POINTS
{ref_pts}

---

CANDIDATE POSTS - MAIN POINTS
{cand_blocks}

---

TASK
Return a JSON array of candidate IDs that cover **similar ideas**
as the reference (≥15%+ content overlap).  Only output the JSON array.
"""

NOVELTY_PROMPT = """
You are an editor who triages AI‑safety research.  
Your job is to judge **how much genuinely NEW intellectual contribution** a
reference article adds relative to prior overlapping work.

---  SCORING RUBRIC ---
•  0 - 20  Identical or near‑identical (e.g. cross‑posted, summary of earlier posts).  
• 21 - 40  Mostly re‑hash or incremental framing; at most a footnote of new insight.  
• 41 - 70  Adds one clear new argument, dataset, empirical result, policy angle, etc.  
• 71 - 90  Several substantive advances or a novel synthesis that changes how the
           topic should be approached or discussed.  
• 91 - 100 Breakthrough content; would matter a lot to most experts following the field.

Use the **lowest score that fits**; be stingy with 95+.  
Ignore style, length, or popularity – focus only on conceptual novelty.

---  OUTPUT FORMAT ---
Return **only** valid minified JSON – no markdown, no code‑fence, no commentary:

{{
  "uniqueness_score": <integer 0‑100>,
  "what_is_new": "<one paragraph ≤ 100 words>"
}}

`what_is_new` must:
  • Describe *exactly* the delta versus the overlaps (not the whole paper).  
  • Be explicit (“Introduces a new empirical estimate of GPU demand…”,  
    “Proposes a formal definition of model deception…”, etc.).  
  • ≤ 100 words.  
  • Make it very readable; this will be shown to news‑feed scrollers deciding whether to read the full article.

--- INPUT DATA  ---
REFERENCE ID {ref_id}
Title: {ref_title}
Full Content:
{ref_content}

OVERLAPPING ARTICLES
{overlap_blocks}
"""

NOVELTY_NO_OVERLAPS_PROMPT = """
You are an editor who triages AI‑safety research.
Your job is to judge **how much genuinely NEW intellectual contribution** an
article adds to the field. In this case, we found no overlapping content in our database.

This suggests the content may be highly novel, but it could also be:
1. Addressing a very niche area we haven't covered
2. Using terminology or framing that differs from similar work
3. The first piece on an emerging topic

---  SCORING RUBRIC ---
•  0 - 20  Identical or near‑identical (e.g. cross‑posted, summary of earlier posts).  
• 21 - 40  Mostly re‑hash or incremental framing; at most a footnote of new insight.  
• 41 - 70  Adds one clear new argument, dataset, empirical result, policy angle, etc.  
• 71 - 90  Several substantive advances or a novel synthesis that changes how the
           topic should be approached or discussed.  
• 91 - 100 Breakthrough content; would matter a lot to most experts following the field.

Err on the side of lower scores when uncertain.

---  OUTPUT FORMAT ---
Return **only** valid minified JSON – no markdown, no code‑fence, no commentary:

{{
  "uniqueness_score": <integer 70‑100>,
  "what_is_new": "<one paragraph ≤ 100 words>"
}}

`what_is_new` must:
  • Describe the apparent novelty based on your analysis.
  • Be explicit about what seems new in the field.
  • ≤ 100 words.
  • Make it very readable; this will be used by news‑feed scrollers to decide if they need to read the full article.
  
--- INPUT DATA  ---
REFERENCE ID {ref_id}
Title: {ref_title}
Full Content:
{ref_content}

NOTE: Our database found no overlapping content with this piece.
"""

# ───── Small helpers ──────────────────────────────────────────
def build_overlap_prompt(ref, cands):
    ref_pts = ref["paragraph_summary"] or ref["sentence_summary"] or ""
    blocks  = [f"- id:{c['id']}\n  {c['paragraph_summary'] or c['sentence_summary'] or ''}"
               for c in cands]
    return OVERLAP_PROMPT.format(ref_pts=ref_pts,
                                 cand_blocks="\n".join(blocks))

def build_novelty_prompt(ref, overlaps):
    ref_content = (ref["full_content_markdown"] or
                   ref["paragraph_summary"]      or
                   ref["sentence_summary"]       or "")
    blocks  = [f"- id:{o['id']}\n  {o['full_content_markdown'] or o['paragraph_summary'] or o['sentence_summary'] or ''}"
               for o in overlaps]
    return NOVELTY_PROMPT.format(ref_id       = ref["id"],
                                 ref_title    = ref["title"],
                                 ref_content  = ref_content,
                                 overlap_blocks="\n".join(blocks))

# ───── Main routine ───────────────────────────────────────────
def main():
    t0   = time.time()
    conn = None
    cur  = None
    processed_count = 0
    try:
        conn = psycopg2.connect(DB_URL)
        cur  = conn.cursor(cursor_factory=extras.RealDictCursor)

        # 1️⃣  newest refs lacking novelty_score
        logging.info("Fetching refs needing novelty scores…")
        cur.execute("""
            SELECT id, title, sentence_summary, paragraph_summary,
                   embedding_full, full_content_markdown
            FROM   content
            WHERE  novelty_score IS NULL
                   AND embedding_full IS NOT NULL
            ORDER  BY published_date DESC NULLS LAST
            LIMIT  %s
        """, (BATCH,))
        refs = cur.fetchall()
        if not refs:
            logging.info("Nothing to back‑fill – all caught up.")
            return

        logging.info("Fetched %d refs to process.", len(refs))

        # Process each ref individually
        for ref in refs:
            rid = ref["id"]
            try:
                logging.info("[%s] analysing…", rid)

                # 2️⃣  nearest neighbours
                cur.execute("""
                    SELECT id, title, sentence_summary, paragraph_summary,
                           full_content_markdown
                    FROM   content
                    WHERE  id <> %s
                    ORDER  BY embedding_full <=> %s
                    LIMIT  %s
                """, (rid, ref["embedding_full"], K_NEIGHBOURS))
                cands = cur.fetchall()

                neighbour_ids = [c['id'] for c in cands]
                logging.info("   → Fetched %d neighbours: %s", len(neighbour_ids), neighbour_ids)

                # 3️⃣  overlap filter
                overlap_ids = openai_json(build_overlap_prompt(ref, cands),
                                           temperature=0.0)
                
                overlap_int_ids = set()
                
                # Handle both {"ids": [...]} and plain [...] formats
                ids_to_process = None
                if isinstance(overlap_ids, dict) and "ids" in overlap_ids:
                    ids_to_process = overlap_ids["ids"]
                elif isinstance(overlap_ids, dict) and "result" in overlap_ids:
                    ids_to_process = overlap_ids["result"]
                elif isinstance(overlap_ids, list):
                    ids_to_process = overlap_ids
                else:
                    logging.warning("   → Unexpected overlap response format: %s", overlap_ids)
                    ids_to_process = []
                
                if isinstance(ids_to_process, list):
                    for item_id in ids_to_process:
                        try:
                            overlap_int_ids.add(int(item_id))
                        except (ValueError, TypeError):
                            logging.warning("   → Could not convert overlap ID '%s' to int. Skipping.", item_id)

                overlaps = [c for c in cands if c["id"] in overlap_int_ids]
                logging.info("   → %d overlaps found", len(overlaps))

                # 4️⃣  novelty score + note
                if not overlaps:
                    logging.info("   → No overlaps found, using specialized prompt")
                    ref_content = (ref["full_content_markdown"] or
                                   ref["paragraph_summary"]      or
                                   ref["sentence_summary"]       or "")
                    nov_prompt = NOVELTY_NO_OVERLAPS_PROMPT.format(
                        ref_id     = rid,
                        ref_title  = ref["title"],
                        ref_content= ref_content
                    )
                    nov = openai_json(nov_prompt,
                                      temperature=0.1)
                else:
                    nov_prompt = build_novelty_prompt(ref, overlaps)
                    nov = openai_json(nov_prompt,
                                      temperature=0.1)

                logging.info("   → Raw LLM JSON output: %s", nov)

                if nov is None:
                    logging.warning("   → LLM call failed or returned invalid JSON for [%s]. Skipping DB update.", rid)
                    continue

                if isinstance(nov, list) and len(nov) == 1 and isinstance(nov[0], dict):
                    nov_data = nov[0]
                elif isinstance(nov, dict):
                    nov_data = nov
                else:
                    nov_data = None

                score = int(nov_data.get("uniqueness_score", 0)) if nov_data else 0
                note  = nov_data.get("what_is_new", "LLM analysis failed.") if nov_data else "LLM analysis failed."

                logging.info("   → score=%3d", score)

                # 5️⃣  write back immediately
                cur.execute(
                    "UPDATE content SET novelty_score=%s, novelty_note=%s WHERE id=%s",
                    (score, note[:1000], rid)
                )
                conn.commit()
                logging.info("   → DB updated successfully for [%s].", rid)
                processed_count += 1

            except Exception as e:
                logging.error("Failed to process ref [%s]: %s", rid, e, exc_info=True)
                if conn:
                    conn.rollback()

        logging.info("Finished processing. Successfully updated %d/%d refs in %.1fs",
                     processed_count, len(refs), time.time() - t0)

    except psycopg2.Error as db_err:
        logging.error("Database error occurred: %s", db_err, exc_info=True)
        if conn:
            conn.rollback()
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e, exc_info=True)
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
