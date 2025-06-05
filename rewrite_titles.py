#!/usr/bin/env python3
"""
backfill_cleaned_titles.py
─────────────────────────
Generate a concise, descriptive, and standardised **cleaned title** for
posts in the `content` table that are still missing one.

The script leaves the original `title` column untouched and writes the
new value to `cleaned_title`. Also generates images and saves them to
`cleaned_image` and the image prompt to `image_prompt`.

Run *once* beforehand:

    ALTER TABLE content
        ADD COLUMN cleaned_title text;
    
    ALTER TABLE content 
        ADD COLUMN image_prompt text;

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
from PIL import Image
import tempfile
import webbrowser
import io
import subprocess
import platform
import csv
from datetime import datetime

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
BATCH          = 1         # rows per execution
# Default mode can be changed here if needed
DEFAULT_MODE   = "titles"

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


def save_image_to_file(image_bytes: bytes, content_id: int, variant_num: int, images_dir: str = "generated_images") -> str:
    """Save image bytes to a file with naming convention: {content_id}_image_{variant_num}.png"""
    os.makedirs(images_dir, exist_ok=True)
    filename = f"{content_id}_image_{variant_num}.png"
    filepath = os.path.join(images_dir, filename)
    
    with open(filepath, 'wb') as f:
        f.write(image_bytes)
    
    return filepath


# ───── Main routine ───────────────────────────────────────────
def main():
    t0 = time.time()
    conn = cur = None
    updated = 0

    parser = argparse.ArgumentParser(description="Generate cleaned titles and/or images for content.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["titles", "images", "images-batch", "upload", "both"],
        default=DEFAULT_MODE,
        help="Operation mode: 'titles' to only generate titles, 'images' to only generate images interactively, 'images-batch' to generate images in batch mode, 'upload' to interactively upload generated images, 'both' to generate both. Default is 'both'."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of articles to process in batch mode (only applies to images-batch mode). Default is 20."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Directory containing generated images (for upload mode). If not specified, will look for the most recent generated_images_* directory."
    )
    args = parser.parse_args()
    mode = args.mode
    batch_size = args.batch_size if mode == "images-batch" else BATCH
    images_dir = args.images_dir
    
    logging.info(f"Running in mode: {mode}")
    if mode == "images-batch":
        logging.info(f"Batch size: {batch_size}")

    # Handle upload mode separately
    if mode == "upload":
        return handle_interactive_upload(images_dir)

    try:
        conn = psycopg2.connect(DB_URL)
        cur  = conn.cursor(cursor_factory=extras.RealDictCursor)

        # Check if image_prompt column exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='content' AND column_name='image_prompt'
        """)
        if not cur.fetchone():
            logging.error("The 'image_prompt' column does not exist in the 'content' table.")
            logging.error("Please run: ALTER TABLE content ADD COLUMN image_prompt text;")
            return

        # 1️⃣  Select rows based on mode
        select_query = ""
        if mode == "titles":
            logging.info("Selecting rows where cleaned_title IS NULL …")
            select_query = """
                SELECT id, title, sentence_summary, paragraph_summary, full_content_markdown, cleaned_title, cleaned_image, image_prompt
                FROM   content
                WHERE  cleaned_title IS NULL
                ORDER  BY published_date DESC NULLS LAST
                LIMIT  %s
            """
        elif mode in ["images", "images-batch"]:
            logging.info("Selecting rows where cleaned_image IS NULL and cleaned_title IS NOT NULL …")
            select_query = """
                SELECT id, title, sentence_summary, paragraph_summary, full_content_markdown, cleaned_title, cleaned_image, image_prompt
                FROM   content
                WHERE  cleaned_image IS NULL AND cleaned_title IS NOT NULL
                ORDER  BY published_date DESC NULLS LAST
                LIMIT  %s
            """
        elif mode == "both":
            logging.info("Selecting rows where cleaned_title IS NULL or cleaned_image IS NULL …")
            select_query = """
                SELECT id, title, sentence_summary, paragraph_summary, full_content_markdown, cleaned_title, cleaned_image, image_prompt
                FROM   content
                WHERE  cleaned_title IS NULL OR cleaned_image IS NULL
                ORDER  BY published_date DESC NULLS LAST
                LIMIT  %s
            """
        
        cur.execute(select_query, (batch_size,))
        rows = cur.fetchall()

        if not rows:
            logging.info("Nothing to do for the selected mode.")
            return
        logging.info("Fetched %d rows", len(rows))

        # Handle batch image generation mode
        if mode == "images-batch":
            return handle_batch_image_generation(rows)

        # 2️⃣  Process individually so errors don't block others (original logic for non-batch modes)
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
            current_image_prompt = row["image_prompt"]

            # These will hold the values to be potentially written to the DB
            final_title_to_write = current_cleaned_title
            final_cdn_url_to_write = current_cdn_url
            final_image_prompt_to_write = current_image_prompt

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

            # --- Stage 2: Image Generation (Interactive Mode) --- 
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
                
                # --- Sub-Stage: Concept Extraction for Image Prompt ---
                concept_extraction_prompt = f"""
                Analyze the following article content and identify the top most important distinct concept presented.
                Then think of a visual metaphor that is radical and can be used to represent the concept.
                Output the concise most important distinct concept (max length of 20 words) and the visual metaphor (max length of 12 words).
                
                ---

                Article:
                
                Title: {final_title_to_write}

                Full Content:
                {full_content}
                """
                logging.info("[%s] Attempting to extract concepts for image generation...", cid)
                
                try:
                    rsp = text_client.chat.completions.create(
                        model = "gpt-4.1",
                        messages = [
                            {"role": "user", "content": concept_extraction_prompt}
                        ],
                        temperature = 0.5,
                        max_tokens = 300
                    )

                    concept_result = rsp.choices[0].message.content.strip()

                    logging.info("   → Concepts extracted for image: %s", concept_result)
                    
                    if concept_result:
                        img_prompt = f""" 
                        Create a raw image of the visual metaphor below:
                        {concept_result}

                        Required Style: Raw, fun, & grungy paper cut-out animation
                        """
                    else:
                        logging.warning("   → LLM failed to generate concepts.")
                except Exception as e:
                    logging.warning("   → LLM concept extraction failed: %s", e)
                    img_prompt = f""" 
                    Create a raw image for this article:
                    Title: {final_title_to_write}
                    Summary: {paragraph_summary}
                    """

                try:
                    # Generate 3 images but DON'T upload yet - keep them local for user selection
                    variant_files: list[tuple[str, bytes]] = []   # (path, raw_bytes)
                    logging.info("   → Generating 3 image variants...")
                    
                    for i in range(3):
                        img_rsp = image_client.images.generate(
                            model="gpt-image-1",
                            prompt=img_prompt,
                            size="1024x1024",
                            quality="medium",
                        )
                        img_b64_data = img_rsp.data[0].b64_json 
                        image_bytes = base64.b64decode(img_b64_data) 
                        
                        # Show the image and store its path and bytes
                        path = _show_temp_image(image_bytes, i+1)
                        variant_files.append((path, image_bytes))
                        logging.info(f"   → Generated and opened image variant {i+1}")
                        
                        # Small delay to prevent overwhelming the system
                        time.sleep(0.5)

                    # Give user time to see all images before prompting
                    logging.info("   → All 3 variants should now be open in your image viewer.")
                    logging.info("   → Please review all variants before making your selection.")
                    
                    # Ask user to choose their favorite
                    choice = input(f"[{cid}] Enter variant # to upload (1-3) or 's' to skip: ").strip()
                    if choice in {"1", "2", "3"}:
                        idx = int(choice) - 1
                        sel_path, sel_bytes = variant_files[idx]

                        # Upload only the chosen variant
                        up_rsp = cloudinary.uploader.upload(
                            io.BytesIO(sel_bytes),
                            folder="ai-safety-feed",
                            public_id=f"{cid}_variant_{choice}"
                        )
                        final_cdn_url_to_write = up_rsp["secure_url"]
                        final_image_prompt_to_write = img_prompt.strip()  # Save the prompt that was used
                        image_generated_this_run = True
                        logging.info("   → Uploaded variant %s: %s", choice, final_cdn_url_to_write)
                    else:
                        logging.info("   → Image selection skipped; nothing uploaded.")
                        
                    # Clean up temporary files
                    for path, _ in variant_files:
                        try:
                            os.unlink(path)
                        except Exception:
                            pass  # Ignore cleanup errors

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
                update_fields.append("image_prompt=%s")
                update_values.append(final_image_prompt_to_write)

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
                        log_message_parts.append(f"Image prompt saved: '{final_image_prompt_to_write[:100]}...'")
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


def handle_batch_image_generation(rows):
    """Handle batch image generation mode - generate all images, save locally, create CSV for selection."""
    logging.info("Starting batch image generation mode...")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    images_dir = f"generated_images_{timestamp}"
    os.makedirs(images_dir, exist_ok=True)
    
    # CSV data
    csv_data = []
    
    # Database connection for saving image prompts
    conn = cur = None
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        
        for row in rows:
            cid = row["id"]
            cleaned_title = row["cleaned_title"] or "No title available"
            full_content = row["full_content_markdown"] or ""
            paragraph_summary = row["paragraph_summary"] or "No paragraph summary available"
            
            # Truncate full content if too long
            if len(full_content) > 50000:
                full_content = full_content[:50000] + "\n\n[rest of the content cut due to size]"
            
            logging.info(f"[{cid}] Generating images for: '{cleaned_title}' ...")
            
            # Extract concepts for image prompt
            concept_extraction_prompt = f"""
            Analyze the following article content and identify the top most important distinct concept presented.
            Then think of a visual metaphor that is radical and can be used to represent the concept.
            Output the concise most important distinct concept (max length of 20 words) and the visual metaphor (max length of 12 words).
            
            ---

            Article:
            
            Title: {cleaned_title}

            Full Content:
            {full_content}
            """
            
            try:
                rsp = text_client.chat.completions.create(
                    model = "gpt-4.1",
                    messages = [
                        {"role": "user", "content": concept_extraction_prompt}
                    ],
                    temperature = 0.5,
                    max_tokens = 300
                )

                concept_result = rsp.choices[0].message.content.strip()
                logging.info(f"   → Concepts extracted: {concept_result}")
                
                if concept_result:
                    img_prompt = f""" 
                    Create a raw image incorporating the visual metaphor below:
                    {concept_result}

                    Required Style: Raw, fun, & grungy paper cut-out animation
                    """
                else:
                    img_prompt = f""" 
                    Create a raw image for this article:
                    Title: {cleaned_title}
                    Summary: {paragraph_summary}
                    """
            except Exception as e:
                logging.warning(f"   → Concept extraction failed: {e}")
                img_prompt = f""" 
                Create a raw image for this article:
                Title: {cleaned_title}
                Summary: {paragraph_summary}
                """

            # Save the image prompt to the database immediately
            try:
                cur.execute(
                    "UPDATE content SET image_prompt = %s WHERE id = %s",
                    (img_prompt.strip(), cid)
                )
                conn.commit()
                logging.info(f"   → Image prompt saved to database")
            except Exception as e:
                logging.warning(f"   → Failed to save image prompt to database: {e}")

            # Generate 3 images
            images_generated = 0
            for i in range(3):
                try:
                    logging.info(f"   → Generating image variant {i+1}/3...")
                    img_rsp = image_client.images.generate(
                        model="gpt-image-1",
                        prompt=img_prompt,
                        size="1024x1024",
                        quality="medium",
                    )
                    img_b64_data = img_rsp.data[0].b64_json 
                    image_bytes = base64.b64decode(img_b64_data)
                    
                    # Save to file with proper naming
                    filepath = save_image_to_file(image_bytes, cid, i+1, images_dir)
                    logging.info(f"   → Saved: {filepath}")
                    images_generated += 1
                    
                    # Small delay to prevent rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logging.warning(f"   → Failed to generate image variant {i+1}: {e}")
            
            # Add to CSV data (only if at least one image was generated)
            if images_generated > 0:
                csv_data.append({
                    'id': cid,
                    'article_title': cleaned_title,
                    'image_prompt': img_prompt.strip(),
                    'image_num': '',  # To be filled by user
                    'images_generated': images_generated
                })
                logging.info(f"   → Generated {images_generated}/3 images for article {cid}")
            else:
                logging.warning(f"   → No images generated for article {cid}")
        
    except psycopg2.Error as db_err:
        logging.error("Database error during batch generation: %s", db_err, exc_info=True)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
    
    # Create CSV file
    csv_filename = f"image_selection_{timestamp}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'article_title', 'image_prompt', 'image_num', 'images_generated']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    logging.info(f"Batch generation complete!")
    logging.info(f"Images saved to: {images_dir}/")
    logging.info(f"CSV created: {csv_filename} (for reference)")
    logging.info(f"")
    logging.info(f"Next step:")
    logging.info(f"Run: python rewrite_titles.py --mode upload --images-dir {images_dir}")
    logging.info(f"")
    logging.info(f"This will show you each article's images interactively and let you select which ones to upload.")


def handle_interactive_upload(images_dir: str = None):
    """Handle interactive upload of generated images - prompt user for each ID, upload selected image, update DB."""
    
    # Find the images directory
    if not images_dir:
        # Look for the most recent generated_images_* directory
        dirs = [d for d in os.listdir('.') if d.startswith('generated_images_')]
        if not dirs:
            logging.error("No generated_images_* directories found. Please specify --images-dir or run images-batch mode first.")
            return
        images_dir = max(dirs)  # Get the most recent (alphabetically last)
        logging.info(f"Using most recent images directory: {images_dir}")
    
    if not os.path.exists(images_dir):
        logging.error(f"Images directory does not exist: {images_dir}")
        return
    
    # Scan for image files and group by content ID
    content_ids = {}
    for filename in os.listdir(images_dir):
        if filename.endswith('.png') and '_image_' in filename:
            # Parse filename: {content_id}_image_{variant_num}.png
            parts = filename.replace('.png', '').split('_image_')
            if len(parts) == 2:
                try:
                    content_id = int(parts[0])
                    variant_num = int(parts[1])
                    
                    if content_id not in content_ids:
                        content_ids[content_id] = []
                    content_ids[content_id].append((variant_num, filename))
                except ValueError:
                    logging.warning(f"Skipping file with unexpected format: {filename}")
    
    if not content_ids:
        logging.error(f"No valid image files found in {images_dir}")
        return
    
    logging.info(f"Found images for {len(content_ids)} content IDs")
    
    # Database connection
    conn = cur = None
    updated = 0
    
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        
        # Process each content ID
        for content_id in sorted(content_ids.keys()):
            variants = sorted(content_ids[content_id])  # Sort by variant number
            
            # Get article title for context
            cur.execute("SELECT cleaned_title, image_prompt FROM content WHERE id = %s", (content_id,))
            row = cur.fetchone()
            article_title = row['cleaned_title'] if row else "Unknown title"
            image_prompt = row['image_prompt'] if row and row['image_prompt'] else "No prompt available"
            
            logging.info(f"\n[{content_id}] Article: '{article_title}'")
            logging.info(f"Image Prompt: '{image_prompt}'")
            logging.info(f"Available variants: {[v[0] for v in variants]}")
            
            # Prompt for selection
            while True:
                choice = input(f"[{content_id}] Which image variant? ({'/'.join(str(v[0]) for v in variants)}/s to skip): ").strip().lower()
                
                if choice == 's':
                    logging.info(f"   → Skipped ID {content_id}")
                    break
                elif choice.isdigit() and int(choice) in [v[0] for v in variants]:
                    variant_num = int(choice)
                    # Find the filename for this variant
                    filename = next(f for v, f in variants if v == variant_num)
                    filepath = os.path.join(images_dir, filename)
                    
                    try:
                        # Upload to Cloudinary
                        logging.info(f"   → Uploading variant {variant_num}...")
                        with open(filepath, 'rb') as f:
                            up_rsp = cloudinary.uploader.upload(
                                f,
                                folder="ai-safety-feed",
                                public_id=f"{content_id}_selected"
                            )
                        cdn_url = up_rsp["secure_url"]
                        
                        # Update database
                        cur.execute(
                            "UPDATE content SET cleaned_image = %s WHERE id = %s",
                            (cdn_url, content_id)
                        )
                        conn.commit()
                        updated += 1
                        
                        logging.info(f"   → Successfully uploaded and updated DB: {cdn_url}")
                        
                        break
                        
                    except Exception as e:
                        logging.error(f"   → Upload/update failed for variant {variant_num}: {e}")
                        break
                else:
                    print(f"Invalid choice. Please enter {'/'.join(str(v[0]) for v in variants)} or 's'")
        
        logging.info(f"\nUpload process complete! Updated {updated} articles.")
        
        # Check if directory is now empty (except for any remaining files)
        remaining_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        if not remaining_files:
            logging.info(f"All images processed. You can delete the directory: {images_dir}")
        else:
            logging.info(f"{len(remaining_files)} images remaining in {images_dir}")
    
    except psycopg2.Error as db_err:
        logging.error("Database error: %s", db_err, exc_info=True)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def _show_temp_image(b64_bytes: bytes, variant_idx: int):
    """Write bytes to a temp *.png, open it non-blocking, return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_v{variant_idx}.png") as tf:
        tf.write(b64_bytes)
    path = tf.name

    try:                                    # try GUI first - non-blocking
        if platform.system() == "Windows":
            # Use os.startfile for Windows - cleaner and more reliable
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", path])
        elif platform.system() == "Linux":
            # Non-blocking xdg-open for Linux
            subprocess.Popen(["xdg-open", path])
        else:
            # Fallback: open in browser (works even over SSH with X-forwarding)
            webbrowser.open(f"file://{path}")
    except Exception:
        # Fallback: open in browser
        webbrowser.open(f"file://{path}")

    return path


if __name__ == "__main__":
    main()
