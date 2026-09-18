#!/usr/bin/env python3
"""
novelty_analysis.py
──────────────────────────
Compute a "uniqueness / novelty" score plus a short
'what-this-adds' note for the newest posts that still lack them,
using **Azure OpenAI**.
"""

# ───── Imports ────────────────────────────────────────────────
import os, sys, time, json, logging, psycopg2
from psycopg2 import extras
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# ───── Runtime settings ───────────────────────────────────────
DB_URL = os.getenv("AI_SAFETY_FEED_DB_URL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"
MODEL = AZURE_OPENAI_DEPLOYMENT
BATCH = 21
K_NEIGHBOURS = 20
MAX_SUMMARY_CHARS = int(os.getenv("NOVELTY_MAX_SUMMARY_CHARS", "1200"))
MAX_REF_CONTENT_CHARS = int(os.getenv("NOVELTY_MAX_REF_CONTENT_CHARS", "12000"))
MAX_OVERLAP_CONTENT_CHARS = int(os.getenv("NOVELTY_MAX_OVERLAP_CONTENT_CHARS", "3000"))
MAX_OVERLAPS_FOR_NOVELTY_PROMPT = int(os.getenv("NOVELTY_MAX_OVERLAPS_FOR_PROMPT", "8"))

if not DB_URL:
    sys.exit("Set AI_SAFETY_FEED_DB_URL")
if not AZURE_OPENAI_API_KEY:
    sys.exit("Set AZURE_OPENAI_API_KEY (or AZURE_API_KEY)")
if not AZURE_OPENAI_ENDPOINT:
    sys.exit("Set AZURE_OPENAI_ENDPOINT")
if not AZURE_OPENAI_DEPLOYMENT:
    sys.exit("Set AZURE_OPENAI_DEPLOYMENT (e.g. gpt-5.6-terra)")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)

# ───── Azure OpenAI client ────────────────────────────────────
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    timeout=180,
)


def extract_response_text(response) -> str:
    """Normalize Azure chat response content into plain text."""
    try:
        if not response or not response.choices:
            return ""
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", "") or "")
                else:
                    parts.append(getattr(part, "text", "") or "")
            return "".join(parts).strip()
    except Exception:
        return ""
    return ""


def openai_json(prompt: str,
                *,
                model: str | None = None,
                temperature: float = 0.15,
                max_tokens: int = 2048):
    """Return parsed JSON (or None) from Azure OpenAI in strict-JSON mode."""
    sys_prompt = (
        "You are a service that MUST return **only** valid minified JSON. "
        "Do not wrap the JSON in markdown or add any commentary."
    )
    raw_text = ""
    try:
        target_model = model or MODEL
        logging.info("Using model: %s", target_model)

        request_kwargs = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        # gpt-5 deployments do not support temperature.
        if temperature is not None and not str(target_model).lower().startswith("gpt-5"):
            request_kwargs["temperature"] = temperature

        rsp = client.chat.completions.create(**request_kwargs)
        raw_text = extract_response_text(rsp)
        return json.loads(raw_text)

    except json.JSONDecodeError as json_err:
        logging.warning("Azure JSON failure - Parsing error: %s. Text was: %s", json_err, raw_text[:200])
        return None
    except Exception as e:
        logging.warning("Azure JSON failure - %s", e)
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
Return a JSON object with an array of candidate IDs that cover **similar ideas**
as the reference (≥15%+ content overlap).

Output format (minified JSON only – no commentary):
{{"ids":[<integer candidate ids>]}}

If there are no overlapping candidates, return:
{{"ids":[]}}
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
    ref_pts = (ref["paragraph_summary"] or ref["sentence_summary"] or "")[:MAX_SUMMARY_CHARS]
    blocks = []
    for c in cands:
        cand_text = (c["paragraph_summary"] or c["sentence_summary"] or "")[:MAX_SUMMARY_CHARS]
        blocks.append(f"- id:{c['id']}\n  {cand_text}")
    return OVERLAP_PROMPT.format(ref_pts=ref_pts,
                                 cand_blocks="\n".join(blocks))

def build_novelty_prompt(
    ref,
    overlaps,
    *,
    max_overlaps: int = MAX_OVERLAPS_FOR_NOVELTY_PROMPT,
    max_ref_chars: int = MAX_REF_CONTENT_CHARS,
    max_overlap_chars: int = MAX_OVERLAP_CONTENT_CHARS,
):
    ref_content = (ref["full_content_markdown"] or
                   ref["paragraph_summary"]      or
                   ref["sentence_summary"]       or "")[:max_ref_chars]
    selected_overlaps = overlaps[:max_overlaps]
    blocks = []
    for o in selected_overlaps:
        overlap_text = (o["full_content_markdown"] or o["paragraph_summary"] or o["sentence_summary"] or "")[:max_overlap_chars]
        blocks.append(f"- id:{o['id']}\n  {overlap_text}")
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
                   embedding_full, full_content_markdown, published_date
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
                # Only compare against work published BEFORE the reference post;
                # otherwise back-filled rows get penalised for later articles.
                if ref.get("published_date"):
                    cur.execute("""
                        SELECT id, title, sentence_summary, paragraph_summary,
                               full_content_markdown
                        FROM   content
                        WHERE  id <> %s
                               AND embedding_full IS NOT NULL
                               AND published_date < %s
                        ORDER  BY embedding_full <=> %s
                        LIMIT  %s
                    """, (rid, ref["published_date"], ref["embedding_full"], K_NEIGHBOURS))
                else:
                    cur.execute("""
                        SELECT id, title, sentence_summary, paragraph_summary,
                               full_content_markdown
                        FROM   content
                        WHERE  id <> %s AND embedding_full IS NOT NULL
                        ORDER  BY embedding_full <=> %s
                        LIMIT  %s
                    """, (rid, ref["embedding_full"], K_NEIGHBOURS))
                cands = cur.fetchall()

                neighbour_ids = [c['id'] for c in cands]
                logging.info("   → Fetched %d neighbours: %s", len(neighbour_ids), neighbour_ids)

                # 3️⃣  overlap filter
                overlap_ids = openai_json(build_overlap_prompt(ref, cands),
                                           temperature=0.0)
                
                # Validate the overlap response strictly: anything malformed leaves the row
                # for a later run instead of falling through to the "no overlaps" prompt.
                if overlap_ids is None or (isinstance(overlap_ids, dict) and "error" in overlap_ids):
                    logging.warning("   → Overlap call failed or reported an error for [%s]: %s. Leaving row for a later run.", rid, overlap_ids)
                    continue
                if isinstance(overlap_ids, dict):
                    ids_raw = overlap_ids.get("ids", overlap_ids.get("result"))
                else:
                    ids_raw = overlap_ids
                if not isinstance(ids_raw, list):
                    logging.warning("   → Unexpected overlap response format for [%s]: %s. Leaving row for a later run.", rid, overlap_ids)
                    continue
                candidate_ids = {c["id"] for c in cands}
                overlap_int_ids = set()
                for item_id in ids_raw:
                    try:
                        value = int(item_id)
                    except (ValueError, TypeError):
                        continue
                    if value in candidate_ids:
                        overlap_int_ids.add(value)
                if ids_raw and not overlap_int_ids:
                    logging.warning("   → Overlap response for [%s] contained no valid candidate IDs: %s. Leaving row for a later run.", rid, ids_raw)
                    continue

                overlaps = [c for c in cands if c["id"] in overlap_int_ids]
                logging.info("   → %d overlaps found", len(overlaps))

                # 4️⃣  novelty score + note
                if not overlaps:
                    logging.info("   → No overlaps found, using specialized prompt")
                    ref_content = (ref["full_content_markdown"] or
                                   ref["paragraph_summary"]      or
                                   ref["sentence_summary"]       or "")[:MAX_REF_CONTENT_CHARS]
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

                    # Retry once with a tighter prompt if context is still too large.
                    if nov is None:
                        logging.info("   → Retrying novelty call with tighter context window")
                        nov_prompt_retry = build_novelty_prompt(
                            ref,
                            overlaps,
                            max_overlaps=3,
                            max_ref_chars=6000,
                            max_overlap_chars=1500,
                        )
                        nov = openai_json(nov_prompt_retry, temperature=0.1)

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

                raw_score = nov_data.get("uniqueness_score") if nov_data else None
                note = nov_data.get("what_is_new") if nov_data else None
                if (isinstance(raw_score, bool) or not isinstance(raw_score, (int, float))
                        or not isinstance(note, str) or not note.strip()):
                    logging.warning("   → Novelty response malformed for [%s]: %s. Leaving row for a later run.", rid, nov)
                    continue
                score = max(0, min(100, int(round(raw_score))))
                note = note.strip()

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
