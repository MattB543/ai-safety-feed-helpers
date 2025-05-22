#!/usr/bin/env python3
"""
backfill_cleaned_titles.py
─────────────────────────
Generate a concise, descriptive, and standardised **cleaned title** for
posts in the `content` table that are still missing one.

The script leaves the original `title` column untouched and writes the
new value to `cleaned_title`.  Run *once* beforehand:

    ALTER TABLE content
        ADD COLUMN cleaned_title text;

Workflow (mirrors backfill_novelty_openai.py):
 1. Pull a batch of recent rows whose `cleaned_title` is NULL.
 2. Ask GPT-4.1 to propose a short, readable title that follows strict
    formatting rules.
 3. Validate the returned JSON and update each row individually so a
    single failure doesn't halt the whole batch.

Environment variables (same as the other backfills):
  • AI_SAFETY_FEED_DB_URL       – PostgreSQL connection string
  • OPEN_AI_FREE_CREDITS_KEY    – OpenAI key for text generation
  • OPEN_AI_WRENLY_KEY         – OpenAI key for image generation
  • CLOUDINARY_URL              – Cloudinary connection string (e.g., cloudinary://key:secret@cloud_name)
"""

# ───── Imports ────────────────────────────────────────────────
import os, sys, time, json, logging, re, psycopg2
from psycopg2 import extras
from openai import OpenAI  # OpenAI Python v1.x client
from dotenv import load_dotenv
import base64 # Added import
import argparse # Added import

# ───── Load environment variables ──────────────────────────────
load_dotenv()  # Load environment variables from .env file

import cloudinary, cloudinary.uploader, requests

# ───── Runtime settings ───────────────────────────────────────
DB_URL         = os.getenv("AI_SAFETY_FEED_DB_URL")
OPENAI_API_KEY = os.getenv("OPEN_AI_FREE_CREDITS_KEY")
OPENAI_WRENLY_KEY = os.getenv("OPEN_AI_WRENLY_KEY")
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")

# Strip potential whitespace from keys
if DB_URL: DB_URL = DB_URL.strip()
if OPENAI_API_KEY: OPENAI_API_KEY = OPENAI_API_KEY.strip()
if OPENAI_WRENLY_KEY: OPENAI_WRENLY_KEY = OPENAI_WRENLY_KEY.strip()
if CLOUDINARY_URL: CLOUDINARY_URL = CLOUDINARY_URL.strip()

MODEL          = "gpt-4.1"   
BATCH          = 5             # rows per execution
# Default mode can be changed here if needed
DEFAULT_MODE   = "both"

# ───── Cloudinary config ───────────────────────────────────────
cloudinary.config(
    cloudinary_url=CLOUDINARY_URL,   # includes cloud_name, api_key, api_secret
    secure=True
)

if not (DB_URL and OPENAI_API_KEY and OPENAI_WRENLY_KEY and CLOUDINARY_URL):
    sys.exit(
        "Ensure the following environment variables are set: "
        "AI_SAFETY_FEED_DB_URL, OPEN_AI_FREE_CREDITS_KEY, OPEN_AI_WRENLY_KEY, and CLOUDINARY_URL"
    )

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s"
)

# ───── OpenAI helper ──────────────────────────────────────────
text_client = OpenAI(api_key=OPENAI_API_KEY)
image_client = OpenAI(api_key=OPENAI_WRENLY_KEY)

def openai_json(prompt: str,
                *,
                model: str | None = None,
                temperature: float = 0.5,
                max_tokens: int = 500):
    """Call GPT-4.1 in strict-JSON mode and return the parsed dict (or None)."""
    sys_prompt = (
        "You MUST return only minified JSON. Do not wrap it in markdown or add commentary."
    )
    target_model = model or MODEL
    try:
        rsp = text_client.chat.completions.create(
            model           = target_model,
            messages        = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": prompt}
            ],
            temperature     = temperature,
            max_tokens      = max_tokens,
            response_format = {"type": "json_object"}
        )
        raw = rsp.choices[0].message.content
        logging.debug("Raw GPT response: %s", raw)
        return json.loads(raw)

    except json.JSONDecodeError as e:
        logging.warning("JSON decode error: %s — text: %.120s…", e, raw)
        return None
    except Exception as e:
        logging.warning("OpenAI call failed: %s", e)
        return None


# ───── Main routine ───────────────────────────────────────────
def main():
    t0 = time.time()
    conn = cur = None
    updated = 0

    parser = argparse.ArgumentParser(description="Generate cleaned titles and/or images for content.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["titles", "images", "both"],
        default=DEFAULT_MODE,
        help="Operation mode: 'titles' to only generate titles, 'images' to only generate images, 'both' to generate both. Default is 'both'."
    )
    args = parser.parse_args()
    mode = args.mode
    logging.info(f"Running in mode: {mode}")

    try:
        conn = psycopg2.connect(DB_URL)
        cur  = conn.cursor(cursor_factory=extras.RealDictCursor)

        # 1️⃣  Select rows based on mode
        select_query = ""
        if mode == "titles":
            logging.info("Selecting rows where cleaned_title IS NULL …")
            select_query = """
                SELECT id, title, sentence_summary, paragraph_summary, full_content_markdown, cleaned_title, cleaned_image
                FROM   content
                WHERE  cleaned_title IS NULL
                ORDER  BY published_date DESC NULLS LAST
                LIMIT  %s
            """
        elif mode == "images":
            logging.info("Selecting rows where cleaned_image IS NULL and cleaned_title IS NOT NULL …")
            select_query = """
                SELECT id, title, sentence_summary, paragraph_summary, full_content_markdown, cleaned_title, cleaned_image
                FROM   content
                WHERE  cleaned_image IS NULL AND cleaned_title IS NOT NULL
                ORDER  BY published_date DESC NULLS LAST
                LIMIT  %s
            """
        elif mode == "both":
            logging.info("Selecting rows where cleaned_title IS NULL or cleaned_image IS NULL …")
            select_query = """
                SELECT id, title, sentence_summary, paragraph_summary, full_content_markdown, cleaned_title, cleaned_image
                FROM   content
                WHERE  cleaned_title IS NULL OR cleaned_image IS NULL
                ORDER  BY published_date DESC NULLS LAST
                LIMIT  %s
            """
        
        cur.execute(select_query, (BATCH,))
        rows = cur.fetchall()

        if not rows:
            logging.info("Nothing to do for the selected mode.")
            return
        logging.info("Fetched %d rows", len(rows))

        # 2️⃣  Process individually so errors don't block others
        for row in rows:
            cid   = row["id"]
            original_title = row["title"] or "" # This is the original, uncleaned title from the DB
            sentence_summary = row["sentence_summary"] or "No sentence summary available"
            paragraph_summary = row["paragraph_summary"] or "No paragraph summary available"
            full_content = row["full_content_markdown"] or ""
            
            # Truncate full content if too long
            if len(full_content) > 50000:
                full_content = full_content[:50000] + "\n\n[rest of the content cut due to size]"

            # Initialize with existing values from DB
            current_cleaned_title = row["cleaned_title"]
            current_cdn_url = row["cleaned_image"]

            # These will hold the values to be potentially written to the DB
            final_title_to_write = current_cleaned_title
            final_cdn_url_to_write = current_cdn_url

            title_generated_this_run = False # True if a new title string is successfully generated
            image_generated_this_run = False # True if a new image URL is successfully generated

            # --- Stage 1: Title Generation --- 
            if mode in ["titles", "both"] and not current_cleaned_title:
                logging.info("[%s] Attempting to generate cleaned title …", cid)
                prompt = f"""
                You are an expert copy-editor for an AI-safety news feed.
                Rewrite the given blog-style *Title* into a **concise, descriptive, readable, clear, and
                standardised cleaned title** (8-14 words, ≤ 100 characters) that lets a
                busy reader instantly grasp the article's subject and key claim.

                Guidelines:
                • Keep essential technical terms / named entities.
                • Drop numbering ("#55", "AI #116"), puns, or personal / niche context 
                • Format the title as lowercase sentence-case headings (capitalize only the first word and any proper nouns).
                • Avoid click-bait
                • Never invent facts — only re-express what the original content contains or implies.
                • If the original title is already concise, descriptive, readable, clear, and follows the guidelines, return the original title.

                Return **only** minified JSON:
                {{ "cleaned_title": "<new title>" }}

                ---
                Original Title: "{original_title}"

                Sentence Summary: "{sentence_summary}"

                Paragraph Summary: "{paragraph_summary}"

                Full Content:
                {full_content}
                """
                result = openai_json(prompt)
                if result and isinstance(result, dict) and result.get("cleaned_title"):
                    generated_title_candidate = result["cleaned_title"].strip()
                    if len(generated_title_candidate) > 100 or len(generated_title_candidate.split()) < 4:
                        logging.warning("   → Generated title '%s' did not meet length rules.", generated_title_candidate)
                    else:
                        final_title_to_write = re.sub(r"\s+", " ", generated_title_candidate).strip('" "')
                        title_generated_this_run = True
                        logging.info("   → New title generated: '%s'", final_title_to_write)
                else:
                    logging.warning("   → LLM failed for title generation for id=%s.", cid)
            elif mode in ["titles", "both"] and current_cleaned_title:
                logging.info("[%s] Title already exists and mode is '%s'. Using existing title: '%s'", cid, mode, current_cleaned_title)
                # final_title_to_write is already current_cleaned_title

            # --- Stage 2: Image Generation --- 
            # Condition: 
            # 1. Mode is 'images' or 'both'.
            # 2. A valid title exists for the image prompt (either pre-existing or generated in this run).
            # 3. EITHER current_cdn_url is None (no image exists) 
            #    OR (mode is 'both' AND a new title was just generated in this run, suggesting a potential need for a new image)
            #    OR (mode is 'images' AND current_cdn_url is None) - this is covered by the first part of OR

            should_generate_image = False
            if mode in ["images", "both"] and final_title_to_write:
                if not current_cdn_url: # No image exists
                    should_generate_image = True
                    logging.info("[%s] No existing image. Attempting to generate image.", cid)
                elif mode == "both" and title_generated_this_run and current_cdn_url:
                    # If in 'both' mode, and a new title was made, and an image already exists, 
                    # we assume the new title might warrant a new image.
                    should_generate_image = True
                    logging.info("[%s] New title generated in 'both' mode. Attempting to regenerate image.", cid)
                elif mode == "images" and not current_cdn_url: # This case is implicitly covered by `if not current_cdn_url`
                    # but to be explicit for mode 'images', we only care if image is missing.
                    logging.info("[%s] Mode is 'images' and no image exists. Attempting to generate image.", cid)
                    should_generate_image = True # Redundant due to first check, but clear
                elif mode == "images" and current_cdn_url:
                    logging.info("[%s] Image already exists and mode is 'images'. Skipping image generation.", cid)
                elif mode == "both" and not title_generated_this_run and current_cdn_url:
                     logging.info("[%s] Mode is 'both', title was not re-generated, and image already exists. Skipping image generation.", cid)

            if should_generate_image:
                logging.info("[%s] Generating image for title: '%s' …", cid, final_title_to_write)
                img_prompt = f""" 
            Can you make an aesthetically pleasing image for the below content, please?

            Don't include any text, I just want visuals. Follow this JSON style guide:

            {{
            "style_profile": {{
                "name": "Mid-Century Analytical Poster",
                "core_concept": "Warm, retro-futuristic screen-print aesthetic inspired by 1930-60 instructional graphics."
            }},

            "palette": {{
                "master_swatches": {{
                "warm_band": ["#E78C40", "#F5AC58", "#FCBE61", "#E99A44", "#D98532", "#C46F29"],
                "cool_accents": ["#214A4C", "#2F5E6C", "#699180", "#4D7777"],
                "neutrals": ["#62423A", "#513726", "#42271A", "#C6B285", "#FCF0D4", "#E9E1C7"]
                }},
                "per_image_guidelines": {{
                "base_hues": "Pick 2-4 from warm_band",
                "accent_hues": "0-2 from cool_accents (keep low-saturation)",
                "neutrals": "Unlimited from neutrals for outlines, shadows & background",
                "max_distinct_hue_families": 6,
                "saturation": "0.4-0.65 (HSB space)",
                "temperature_bias": "≥70 % warm hues overall"
                }}
            }},

            "texture_finish": {{
                "grain_overlay": {{
                "type": ["fine paper noise", "subtle halftone dots"],
                "intensity_range": [0.15, 0.30],
                "blend_mode": "multiply"
                }},
                "edge_quality": "soft vector anti-alias",
                "shadow_style": "single hard step; direction may vary per image"
            }},

            "shape_language": {{
                "detail_level": "ultra-minimal (flat fill + 1 interior shadow)",
                "source_icons_from": "NOUNS in the article summary - NOT generic 'AI' imagery.",
                "outlines": "optional; 1-2 px, same neutral for entire piece",
                "silhouette_priority": true
                "negative_space_min": "≥40 % of canvas"
            }},

            "composition": {{
                "negative_space": "reserve ≥40 % of canvas for breathable sky/paper/background/etc.",
                "focal_object": "Exactly ONE hero symbol / design derived from the article."
                "border": "None - no border, no padding, no matting"
            }},

            "lighting": {{
                "background_gradient": ["linear (bottom-dark → top-light)", "radial (center-light)"],
                "highlight_source": "behind or above main subject",
                "shadow_passes": 1
            }},

            "tone_mood": {{
                "keywords_pool": [
                "retro-futuristic",
                "optimistic progress",
                "academic & intellectual rigor",
                "approachable warmth"
                ]
            }},

            "historical_influences": {{
                "primary": "WPA & scientific posters 1930-1960",
                "secondary": ["ligne claire comics", "flat-design revival circa 2019"]
            }},

            "technical_specs": {{
                "color_depth": "8-12 swatches per image (indexing/quantisation allowed)",
                "dpi_for_print": 300
            }},

            "workflow": [
                "Vector: sketch composition, block shapes with chosen palette subset.",
                "Add single hard-edge shadow per object; vary light direction image-to-image.",
                "Export to raster; overlay grain/halftone (intensity 15-30 %).",
                "Subtly color-grade highlights & shadows toward vintage warmth."
            ],

            "variation_levers": {{
                "palette_subset": "Swap one warm base hue or accent between pieces.",
                "grain_intensity": "Tweak within 0.15-0.30 range.",
                "composition_template": "Rotate among rule-of-thirds, central emblem, diagonal sweep.",
                "lighting_angle": "Change gradient or radial center per piece."
            }},

            "do_dont": {{
                "do": [
                "Limit cool accents to <20 % of canvas area.",
                "Ensure strong silhouettes and clear focal hierarchy.",
                "Vary subject matter and layout while retaining palette & texture DNA."
                ],
                "dont": [
                "Use modern soft drop-shadows or radial blurs.",
                "Introduce pure #000000 or #FFFFFF; always offset slightly.",
                "Exceed six distinct hue families in any single piece."
                ]
            }}
            }}


            The goal of the image is to visually represent the specific content of the article below while still being aesthetically pleasing and following the style guide. Remember, no text in the image. It must look good when scaled down to 200 px by 200 px (so clean and simple).

            ---

            Article:
            Title: {final_title_to_write}
            Summary: {paragraph_summary}
            """

                try:
                    img_rsp = image_client.images.generate(
                        model="gpt-image-1",
                        prompt = img_prompt,
                        size   = "1024x1024",
                        quality= "high",
                    )
                    img_b64_data = img_rsp.data[0].b64_json 
                    image_bytes = base64.b64decode(img_b64_data) 

                    up_rsp = cloudinary.uploader.upload(image_bytes, folder="ai-safety-feed")
                    final_cdn_url_to_write = up_rsp["secure_url"]
                    image_generated_this_run = True
                    logging.info("   → New image generated and uploaded: %s", final_cdn_url_to_write)

                except Exception as e:
                    logging.warning("   → Image generation/upload failed: %s", e)
            
            # --- Stage 3: Database Update --- 
            update_fields = []
            update_values = []

            if title_generated_this_run:
                update_fields.append("cleaned_title=%s")
                update_values.append(final_title_to_write)
            
            if image_generated_this_run:
                update_fields.append("cleaned_image=%s")
                update_values.append(final_cdn_url_to_write)

            if update_fields: # If there's anything to update
                update_query_parts = ", ".join(update_fields)
                update_query = f"UPDATE content SET {update_query_parts} WHERE id=%s"
                update_values.append(cid) 
                try:
                    cur.execute(update_query, tuple(update_values))
                    conn.commit()
                    updated += 1
                    log_message_parts = []
                    if title_generated_this_run:
                        log_message_parts.append(f"Title updated: '{final_title_to_write}'")
                    if image_generated_this_run:
                        log_message_parts.append(f"Image updated: {final_cdn_url_to_write}")
                    logging.info(f"   → DB updated for id={cid}: {'; '.join(log_message_parts)}")
                except Exception as db_err:
                    conn.rollback()
                    logging.error("DB update failed for id=%s: %s", cid, db_err)
            else:
                logging.info("[%s] No new data generated to update in DB for this item.", cid)

        logging.info("Done — processed %d items, updated %d rows in %.1fs",
                     len(rows), updated, time.time() - t0)

    except psycopg2.Error as db_err:
        logging.error("Database error: %s", db_err, exc_info=True)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
