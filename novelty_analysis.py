#!/usr/bin/env python3
"""
backfill_novelty.py
───────────────────
Compute a "uniqueness / novelty" score plus a short
'what-this-adds' note for the newest posts that still lack them.
"""

# ───── Imports ────────────────────────────────────────────────
import os, sys, time, json, logging, psycopg2
from psycopg2 import extras
from google import genai as genai           # ← same package as your other script
from google.genai import types
from datetime import datetime

# ───── Runtime settings ───────────────────────────────────────
DB_URL         = os.getenv("AI_SAFETY_FEED_DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL          = "gemini-2.5-flash-preview-04-17"
BATCH          = 20          # rows to back-fill each run
K_NEIGHBOURS   = 20      # vector recall set size

if not (DB_URL and GEMINI_API_KEY):
    sys.exit("Set AI_SAFETY_FEED_DB_URL and GEMINI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)

# ───── Gemini client (same flow as your other script) ─────────
gclient = genai.Client(api_key=GEMINI_API_KEY)

def gemini_json(prompt: str,
                *,
                model: str | None = None,
                temperature: float = 0.15,
                max_tokens: int = 300000):
    """Return parsed JSON (or None) using strict JSON mode."""
    cfg = types.GenerateContentConfig(
        temperature        = temperature,
        max_output_tokens  = max_tokens,
        response_mime_type = "application/json",
    )
    try:
        target_model = model if model else MODEL
        logging.info("Using model: %s", target_model)
        rsp = gclient.models.generate_content(
            model    = target_model,
            contents = prompt,          # ← plain string is fine
            config   = cfg
        )
        raw_text = rsp.text
        logging.info("Raw Gemini response text: %s", raw_text)
        
        # Find the first '[' and last ']' to extract the JSON part
        start = raw_text.find('[')
        end = raw_text.rfind(']')
        
        if start != -1 and end != -1 and start < end:
            json_text = raw_text[start:end+1]
            return json.loads(json_text)
        else:
            # If no brackets found, try parsing the whole text (for dict responses)
            # This handles the NOVELTY_PROMPT case which returns a dict
            try:
                 return json.loads(raw_text)
            except json.JSONDecodeError:
                 # If it's not a dict either, try finding '{' and '}'
                 start_curly = raw_text.find('{')
                 end_curly = raw_text.rfind('}')
                 if start_curly != -1 and end_curly != -1 and start_curly < end_curly:
                    json_text_curly = raw_text[start_curly:end_curly+1]
                    return json.loads(json_text_curly)
                 else:
                     logging.warning("Gemini JSON failure - Could not find valid JSON array or object boundaries.")
                     return None
            
    except json.JSONDecodeError as json_err:
        logging.warning(f"Gemini JSON failure - Parsing error: {json_err}. Text was: {raw_text[:200]}...")
        return None
    except Exception as e:
        logging.warning("Gemini JSON failure - %s", e)
        return None

# ───── Prompt templates (unchanged) ───────────────────────────
OVERLAP_PROMPT = """You are an expert AI-safety editor.

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
You are an editor who triages AI-safety research.  
Your job is to judge **how much genuinely NEW intellectual contribution** a
reference article adds relative to prior overlapping work.

---  SCORING RUBRIC ---
•  0 - 20  Identical or near-identical (e.g. cross-posted, summary of earlier posts).  
• 21 - 40  Mostly re-hash or incremental framing; at most a footnote of new insight.  
• 41 - 70  Adds one clear new argument, dataset, empirical result, policy angle, etc.  
• 71 - 90  Several substantive advances or a novel synthesis that changes how the
           topic should be approached or discussed.  
• 91 - 100 Breakthrough content; would matter a lot to most experts following the field.

Use the **lowest score that fits**; be stingy with 95+.  
Ignore style, length, or popularity - focus only on conceptual novelty.

---  OUTPUT FORMAT ---
Return **only** valid minified JSON - no markdown, no code-fence, no commentary:

{{
  "uniqueness_score": <integer 0-100>,
  "what_is_new": "<one paragraph ≤ 100 words>"
}}

`what_is_new` must:
  • Describe *exactly* the delta versus the overlaps (not the whole paper).  
  • Be explicit (“Introduces a new empirical estimate of GPU demand…”,  
    “Proposes a formal definition of model deception…”, etc.).  
  • ≤ 100 words.
  • Make it very readable, this will be used by newsfeed scrollers to help them decide if they need to read the full article.

--- INPUT DATA  ---
REFERENCE ID {ref_id}
Title: {ref_title}
Full Content:
{ref_content}

OVERLAPPING ARTICLES
{overlap_blocks}
"""

# Add a new prompt template for the zero-overlaps case
NOVELTY_NO_OVERLAPS_PROMPT = """
You are an editor who triages AI-safety research.
Your job is to judge **how much genuinely NEW intellectual contribution** an
article adds to the field. In this case, we found no overlapping content in our database.

This suggests the content may be highly novel, but it could also be:
1. Addressing a very niche area we haven't covered
2. Using terminology or framing that differs from similar work
3. The first piece on an emerging topic

---  SCORING RUBRIC ---
•  0 - 20  Identical or near-identical (e.g. cross-posted, summary of earlier posts).  
• 21 - 40  Mostly re-hash or incremental framing; at most a footnote of new insight.  
• 41 - 70  Adds one clear new argument, dataset, empirical result, policy angle, etc.  
• 71 - 90  Several substantive advances or a novel synthesis that changes how the
           topic should be approached or discussed.  
• 91 - 100 Breakthrough content; would matter a lot to most experts following the field.

Err on the side of lower scores when uncertain.

---  OUTPUT FORMAT ---
Return **only** valid minified JSON - no markdown, no code-fence, no commentary:

{{
  "uniqueness_score": <integer 70-100>,
  "what_is_new": "<one paragraph ≤ 100 words>"
}}

`what_is_new` must:
  • Describe the apparent novelty based on your analysis
  • Be explicit about what seems new in the field
  • ≤ 100 words
  • Make it very readable, this will be used by newsfeed scrollers to help them decide if they need to read the full article.
  
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
    ref_content = ref["full_content_markdown"] or ref["paragraph_summary"] or ref["sentence_summary"] or ""
    blocks  = [f"- id:{o['id']}\n  {o['full_content_markdown'] or o['paragraph_summary'] or o['sentence_summary'] or ''}"
               for o in overlaps]
    return NOVELTY_PROMPT.format(ref_id   = ref["id"],
                                 ref_title= ref["title"],
                                 ref_content = ref_content,
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
        logging.info("Fetching refs needing novelty scores...")
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
            logging.info("Nothing to back-fill - all caught up.")
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

                # Log the IDs of the fetched neighbours
                neighbour_ids = [c['id'] for c in cands]
                logging.info("   → Fetched %d neighbours: %s", len(neighbour_ids), neighbour_ids)

                # 3️⃣  overlap filter
                overlap_ids = gemini_json(build_overlap_prompt(ref, cands),
                                          temperature=0.0, max_tokens=300000)
                # Convert API string IDs to integers for comparison with database IDs
                overlap_int_ids = set()
                if isinstance(overlap_ids, list):
                    for item_id in overlap_ids:
                        try:
                            overlap_int_ids.add(int(item_id))
                        except (ValueError, TypeError):
                            logging.warning(f"   → Could not convert overlap ID '{item_id}' to int. Skipping.")

                overlaps = [c for c in cands if c["id"] in overlap_int_ids]

                logging.info("   → %d overlaps found", len(overlaps))

                # 4️⃣  novelty score + note
                if not overlaps:
                    logging.info("   → No overlaps found, using specialized prompt")
                    ref_content = ref["full_content_markdown"] or ref["paragraph_summary"] or ref["sentence_summary"] or ""
                    nov_prompt = NOVELTY_NO_OVERLAPS_PROMPT.format(
                        ref_id=rid,
                        ref_title=ref["title"],
                        ref_content=ref_content
                    )
                    nov = gemini_json(nov_prompt,
                                     model="gemini-2.5-pro-preview-03-25", # Consider if a different model is needed here
                                     temperature=0.1, max_tokens=300000)
                else:
                    # Use existing logic when overlaps are found
                    nov_prompt = build_novelty_prompt(ref, overlaps)
                    nov = gemini_json(nov_prompt,
                                      model="gemini-2.5-pro-preview-03-25",
                                      temperature=0.1, max_tokens=300000)

                # Log the raw JSON response from Gemini
                logging.info("   → Raw LLM JSON output: %s", nov)

                # Skip update if LLM call failed
                if nov is None:
                    logging.warning("   → LLM call failed or returned invalid JSON for [%s]. Skipping DB update.", rid)
                    continue # Skip to the next ref

                # Handle case where LLM wraps the dict in a list
                if isinstance(nov, list) and len(nov) == 1 and isinstance(nov[0], dict):
                    nov_data = nov[0]
                elif isinstance(nov, dict):
                    nov_data = nov
                else:
                    nov_data = None # Or {} if preferred

                score = int(nov_data.get("uniqueness_score", 0)) if nov_data else 0
                note  = nov_data.get("what_is_new", "LLM analysis failed.") if nov_data else "LLM analysis failed."

                logging.info("   → score=%3d", score)

                # 5️⃣  write back immediately
                cur.execute(
                    "UPDATE content SET novelty_score=%s, novelty_note=%s WHERE id=%s",
                    (score, note[:1000], rid) # truncate note defensively
                )
                conn.commit() # Commit after each successful update
                logging.info("   → DB updated successfully for [%s].", rid)
                processed_count += 1

            except Exception as e:
                logging.error("Failed to process ref [%s]: %s", rid, e, exc_info=True)
                if conn: conn.rollback() # Rollback the transaction for this specific item
                # Continue to the next ref

        logging.info("Finished processing. Successfully updated %d/%d refs in %.1fs",
                     processed_count, len(refs), time.time()-t0)

    except psycopg2.Error as db_err:
        # Catch DB connection or initial fetch errors
        logging.error("Database error occurred: %s", db_err, exc_info=True)
        if conn: conn.rollback()
    except Exception as e:
        # Catch other unexpected errors (e.g., setup)
        logging.error("An unexpected error occurred: %s", e, exc_info=True)
        if conn: conn.rollback()
    finally:
        if cur: cur.close()
        if conn: conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
