#!/usr/bin/env python3
import requests
import json
from datetime import datetime, timedelta, timezone, date
import re
from markdownify import markdownify # Add markdownify import
import os # Add os import for API key and DB URL
import sys # Add sys import for exiting on critical errors
from google import genai as genai # Use alias to avoid potential conflicts
from google.genai import types
import psycopg2 # Add psycopg2 import
from psycopg2 import extras # Import extras for batch insertion
from dotenv import load_dotenv # Add dotenv import
from pgvector.psycopg2 import register_vector # Add pgvector import
import feedparser # Add feedparser import
from urllib.parse import urlparse # Add urlparse import
from bs4 import BeautifulSoup # Add BeautifulSoup import
import time # Add time import
import logging # Add logging import
from openai import OpenAI
from openai import APIError, RateLimitError # Optional: for more specific error handling
from collections import deque # Add deque import

# --- Environment Variables ---
load_dotenv() # Load .env file BEFORE accessing env vars

# --- Top Level Buffer ---
SKIPPED_BUFFER = deque()   # (post_id, title_norm, url)

# --- Helper for title normalization ---
def normalise_title(t: str) -> str:
    """Normalize title: lowercase, replace multiple spaces with single, strip leading/trailing."""
    if not isinstance(t, str): return "" # Handle non-string input
    return re.sub(r'\s+', ' ', t).strip().lower()

def buffer_skip(post_id, title, url):
    """Add a post to the skip buffer, avoiding duplicates."""
    title_norm = normalise_title(title or "")
    # Only append if this title_norm isn't already in the buffer
    if title_norm not in {t for _, t, _ in SKIPPED_BUFFER}:
        SKIPPED_BUFFER.append(
            (str(post_id) if post_id else "N/A",
             title_norm,
             url)
        )

# Setup logging early, before any potential logging calls
# Use INFO level by default, adjust if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress overly verbose logs from underlying libraries if desired
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING) # Added for requests/urllib3 noise
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING) # Suppress OpenAI logs if needed

# --- Essential Environment Variables ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DATABASE_URL   = os.environ.get("AI_SAFETY_FEED_DB_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # <<< ADD THIS

# --- Initial Checks ---
# Use logging instead of print for critical errors before exiting
if not GEMINI_API_KEY:
    logging.critical("CRITICAL ERROR: GEMINI_API_KEY environment variable not set. Cannot perform analysis.")
    sys.exit(1) # Exit if API key is missing

if not DATABASE_URL:
    logging.critical("CRITICAL ERROR: AI_SAFETY_FEED_DB_URL environment variable not set. Cannot connect to database.")
    sys.exit(1) # Exit if DB URL is missing

if not OPENAI_API_KEY: # <<< ADD THIS CHECK
    logging.critical("CRITICAL ERROR: OPENAI_API_KEY environment variable not set. Cannot generate embeddings.")
    sys.exit(1) # Exit if OpenAI key is missing

# --- Constants ---
BATCH_SIZE = 1 # Size for batch database inserts

# Define the cutoff date (inclusive) - posts before this date are ignored
CUTOFF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)

# List of Substack feeds
SUBSTACK_FEEDS = [
    "https://agifriday.substack.com",
    "https://aifrontiersmedia.substack.com",
    "https://www.safeai.news",
    "https://newsletter.safe.ai",
    "https://www.astralcodexten.com", # Note: Changed from .substack.com to .com
    "https://thezvi.substack.com",
    "https://oliverpatel.substack.com",
    "https://epochai.substack.com",
    "https://artificialintelligenceact.substack.com",
    "https://www.hyperdimensional.co",
    "https://joecarlsmith.substack.com",
    "https://milesbrundage.substack.com",
    "https://newsletter.mlsafety.org",
    "https://aligned.substack.com",
    "https://helentoner.substack.com",
]

# Mapping for friendly source names based on Substack host/slug
SUBSTACK_SOURCE_NAMES = {
    "agifriday.substack.com"           : "AGI Friday",
    "aifrontiersmedia.substack.com"    : "AI Frontiers",
    "safeai.news"                      : "AI Safety & Governance Newsletter",
    "newsletter.safe.ai"               : "AI Safety Newsletter", 
    "astralcodexten.com"               : "Astral Codex Ten", 
    "thezvi.substack.com"              : "Don't Worry About the Vase",
    "oliverpatel.substack.com"         : "Enterprise AI Governance",
    "epochai.substack.com"             : "Epoch AI",
    "artificialintelligenceact.substack.com": "The EU AI Act Newsletter",
    "hyperdimensional.co"              : "Hyperdimensional",
    "joecarlsmith.substack.com"        : "Joe Carlsmith's Substack",
    "milesbrundage.substack.com"       : "Miles's Substack",
    "newsletter.mlsafety.org"          : "ML Safety Newsletter",
    "aligned.substack.com"             : "Musings on the Alignment Problem",
    "helentoner.substack.com"          : "Rising Tide",
}

# --- Initialize API Clients --- # <<< ADD THIS SECTION
logging.info("Initializing API clients...")
openai_client = None # Initialize to None
try:
    # Gemini Client is initialized implicitly within call_gemini_api

    # OpenAI Client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("OpenAI client initialized.")
except Exception as e:
    logging.critical(f"CRITICAL ERROR: Failed to initialize OpenAI client: {e}", exc_info=True)
    # Decide if you want to exit if OpenAI client fails, or continue without embeddings
    # sys.exit(1) # Uncomment to exit if OpenAI client fails
logging.info("API clients initialized successfully.")

# --- Database Columns and Pre-computed INSERT statement ---
# ================================================================
#                      Database Configuration
# ================================================================
# Define the columns in the 'content' table that we will insert into
# IMPORTANT: Ensure this order matches the data tuple created later.
# 'title_norm' is excluded as it's a generated column in the DB.
DB_COLS = (
    "source_url", "title", "source_type", "authors", "published_date",
    "topics", "score", "image_url", "sentence_summary", "paragraph_summary",
    "key_implication", "full_content", "full_content_markdown",
    "comment_count", "cluster_tag",
    "embedding_short", "embedding_full" # <<< ADD THESE TWO
)
# Calculate the number of columns dynamically
NUM_DB_COLS = len(DB_COLS)
# Pre-compute the INSERT SQL statement for efficiency
# The f-string logic automatically adjusts the number of '%s' placeholders
INSERT_SQL = f"""
INSERT INTO content ({', '.join(DB_COLS)})
VALUES ({', '.join(['%s'] * NUM_DB_COLS)})
ON CONFLICT (title_norm) DO NOTHING;
"""

# --- Helper SQL & function for skipping posts ---
SKIP_INSERT_SQL = """
INSERT INTO skipped_posts (post_id, title_norm, source_url)
VALUES (%s, %s, %s)
ON CONFLICT (title_norm) DO NOTHING;
"""

def record_skip(cur, post_id: str, title_norm: str, source_url: str | None):
    """
    Insert one row into skipped_posts and keep the caller side-effect-free
    (no commit here – caller decides when to commit or roll back).
    """
    cur.execute(SKIP_INSERT_SQL,
                (str(post_id) if post_id else "N/A",
                 str(title_norm),
                 source_url))

# --- Helper function for Gemini API calls ---
def call_gemini_api(prompt, model_name="gemini-2.5-pro-preview-03-25"): # Removed client parameter
    """Calls the Gemini API with the given prompt and model."""
    if not GEMINI_API_KEY:
        logging.warning("Gemini API call skipped: GEMINI_API_KEY not set.")
        return "Analysis skipped (missing API key)."

    try:
        # Instantiate the client inside the function (legacy way)
        # Consider initializing the client once globally if performance is critical
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content( # Use client.models.generate_content
            model=model_name, # Specify model name here
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        # Extract text (same logic as original summarize_text/user provided example)
        if hasattr(response, 'text'):
            result = response.text.strip()
        elif hasattr(response, 'parts') and response.parts:
            result = "".join(part.text for part in response.parts).strip()
        else:
            logging.warning(f"Warning: Unexpected Gemini API response structure: {response}")
            result = "Error: Could not extract text from API response."

        # Basic check for empty or problematic results
        if not result or result.startswith("Error:"):
             logging.warning(f"Gemini API potentially problematic result: '{result[:100]}...'") # Log problematic result concisely
             # Fallback or return the potentially problematic result

        return result

    except types.generation_types.BlockedPromptException as e: # More specific error
        logging.error(f"Error: Gemini API call failed due to blocked prompt: {e}")
        return f"Error: Analysis blocked due to prompt content."
    except types.generation_types.StopCandidateException as e: # More specific error
        logging.error(f"Error: Gemini API call failed due to stop candidate: {e}")
        return f"Error: Analysis stopped unexpectedly by the model."
    except Exception as e: # Catch other potential API errors
        logging.error(f"Error during Gemini API call: {e}")
        # Optionally log the prompt or part of it for debugging
        # logging.debug(f"Prompt that caused error (first 100 chars): {prompt[:100]}...")
        return f"Error during analysis: {e}"

# --- Analysis Functions ---

def summarize_text(text_to_summarize):
    """Summarizes the input text using the Gemini API (1-2 sentences)."""
    if not text_to_summarize or text_to_summarize.isspace():
        return "Content was empty."

    prompt = f"Summarize the following AI safety content in 2 concise sentences (maximum 50 words). Focus on the core argument, key insight, or main conclusion rather than methodology. Use clear, accessible language while preserving technical accuracy. The summary should be very readable and should help readers quickly understand what makes this content valuable or interesting and decide if they want to read more.\\n\\nContent to summarize:\\n{text_to_summarize}"
    return call_gemini_api(prompt) # Removed client argument

def generate_paragraph_summary(text_to_summarize):
    """Generates a detailed paragraph summary using the Gemini API."""
    if not text_to_summarize or text_to_summarize.isspace():
        return "Content was empty."

    prompt = f"""
Generate a structured summary of the following AI safety content so the reader can quickly understand the main points. The summary should consist of:

1. A brief 1-sentence introduction highlighting the main point.
2. 3-5 bullet points covering key arguments, evidence, or insights. Format EACH bullet point as:
   * **Key concept or term**: Explanation or elaboration of that point.
3. A brief 1-sentence conclusion with the author's recommendation or final thoughts.

---

Rules:
- Make each bullet point concise (1 sentence) and focus on one distinct idea.
- Bold only the key concept at the start of each bullet, not entire sentences.
- This format should help readers quickly scan and understand the core content.
- Only output the summary itself (don't include 'Summary:' or anything else).
- Use markdown to format the bullet points and to improve readability with bolding and italics.
- Include a double line break after the introduction and before the conclusion.

Content to summarize:
{text_to_summarize}
"""
    return call_gemini_api(prompt) # Removed client argument

def generate_key_implication(text_to_analyze):
    """Identifies the single most important logical consequence using the Gemini API."""
    if not text_to_analyze or text_to_analyze.isspace():
        return "Content was empty."

    prompt = f"""
Based on the AI safety content below, identify the single most important logical consequence or implication in one concise sentence (25-35 words). Focus on:

- What change in thinking, strategy, or priorities follows from accepting this content's conclusions
- How this might alter our understanding of AI safety or governance approaches
- A specific actionable insight rather than a general statement of importance
- The "so what" that would matter to an informed AI safety community member

The implication should represent a direct consequence of the content's argument, not simply restate the main point.

Content to analyze:
{text_to_analyze}
"""
    return call_gemini_api(prompt) # Removed client argument

def generate_cluster_tag(title, tags_list, content_markdown):
    """
    Generates a cluster and canonical tags using the Gemini API.

    Returns:
        dict: Parsed JSON like {"cluster": "...", "tags": ["..."]} on success.
        dict: {"error": "Reason string"} on failure (API error, empty content, parsing error).
    """
    if not content_markdown or content_markdown.isspace():
        return {"error": "Content was empty."} # Return error dict
    if not title:
        title = "Unknown" # Provide a default if title is missing
    if not tags_list:
        tags_list = ["Unknown"] # Provide a default if tags are missing

    prompt = f"""
You are the "AI-Safety-Tagger"—an expert taxonomist for an AI-safety news feed.

---  TASK  ---
Given one blog-style post, do BOTH of the following:

1. **Pick exactly one "Cluster"** that best captures the *main theme*
   (see the list of Clusters below).

2. **Choose 1 to 4 "Canonical Tags"** from the same list that most precisely
   describe the post.
   • Tags *must* come from the taxonomy.
   • Prefer the most specific tags that materially help the reader; skip
     generic or redundant ones.
   • A tag may be selected even if it appears only in the "Synonyms"
     column—use its Canonical form in your answer.

Return your answer as valid JSON, with this schema:

{{
  "cluster": "<one Cluster name>",
  "tags": ["<Canonical tag 1>", "... up to 4"]
}}

Do not output anything else.

--- INPUT ---

Title:
{title}

Original author-supplied tags (may be noisy or missing):
{tags_list}

Markdown body:
{content_markdown}

--- TAXONOMY ---

The format is:
• Cluster
- Canonical tag (Synonyms; separated by "")

• Core AI Safety & Alignment
- AI alignment (Human alignment)
- Existential risk (X-risk)
- Threat models (AI) (AI threat models)
- Interpretability (Interpretability (ML & AI); Transparency)
- Inner alignment
- Outer alignment
- Deceptive alignment
- Eliciting latent knowledge (ELK)
- Robustness (Adversarial robustness)
- Alignment field-building (AI alignment field-building)
- Value learning (Preference learning; Alignment via human values)

• AI Governance & Policy
- AI governance (GovAI)
- Compute governance (GPU export controls; Chip governance)
- AI regulation (Regulation)
- Standards & auditing (Safety standards; Red-teaming)
- Responsible scaling (Scaling policies; RSF)
- International coordination (Geopolitics)
- Slowing down AI (Slow takeoff; Pause AI)
- Open-source models (Open-source LLMs)
- Policy (Public policy (generic))
- Compute controls (Hardware throttling)

• Technical ML Safety
- Reinforcement learning (RL)
- Human feedback (RLHF; RLAIF)
- Model editing (Model surgery)
- Scalable oversight (Debate; Tree-of-thought)
- CoT alignment (CoT alignment)
- Scaling laws
- Benchmarks & evals (Safety benchmarks)
- Mechanistic interpretability
- Value decomposition (Shard theory)

• Forecasting & World Modeling
- World modeling
- Forecasting (Quantitative forecasting)
- Prediction markets

• Biorisk & Other GCRs
- Biorisk (Biosecurity; Pandemic preparedness)
- Nuclear risk (Nuclear war; Nuclear winter)
- Global catastrophic risk (GCR)

• Effective Altruism & Meta
- Cause prioritization
- Effective giving
- Career choice (Career planning)
- Community building (Building effective altruism)
- Field-building (AI)
- Epistemics & rationality (Rationality)

• Philosophy & Foundations
- Decision theory (CDT; EDT; UDT)
- Moral uncertainty
- Population ethics
- Agent foundations (Agent foundations research)
- Value drift
- Info hazards (Information hazards)

• Org-specific updates
- Anthropic
- OpenAI
- DeepMind
- Meta
- ARC (Alignment Research Center)

---

Remember: return only JSON with "cluster" and "tags".
"""
    raw_response = call_gemini_api(prompt) # Removed client argument

    # Check if the API call itself returned an error string
    if raw_response.startswith("Error:") or raw_response.startswith("Analysis skipped"):
        return {"error": raw_response}

    # Clean the response: remove markdown fences and trim whitespace
    cleaned_response = raw_response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[len("```json"):].strip()
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-len("```")].strip()

    # Try parsing the cleaned string as JSON
    try:
        parsed_json = json.loads(cleaned_response)
        # Basic validation of structure
        if isinstance(parsed_json, dict) and "cluster" in parsed_json and "tags" in parsed_json and isinstance(parsed_json["tags"], list):
            return parsed_json
        else:
            logging.warning(f"Warning: Parsed JSON from cluster tag API has unexpected structure: {parsed_json}")
            return {"error": "Parsed JSON has unexpected structure"}
    except json.JSONDecodeError as e:
        logging.error(f"Error: Failed to parse JSON from cluster tag API response. Error: {e}")
        logging.debug(f"Cleaned response was: '{cleaned_response}'")
        return {"error": f"Failed to parse JSON response: {e}"}
    except Exception as e: # Catch other potential errors during parsing/validation
        logging.error(f"Error: Unexpected error processing cluster tag API response: {e}")
        return {"error": f"Unexpected error processing cluster tag response: {e}"}


# --- OpenAI Embedding Helper ---
def generate_embeddings(openai_client, short_text: str, full_text: str, model="text-embedding-3-small") -> tuple[list[float] | None, list[float] | None]:
    """
    Generates short and full embeddings for the given texts using OpenAI.

    Args:
        openai_client: Initialized OpenAI client.
        short_text: Text to embed for the 'short' version (e.g., title).
        full_text: Text to embed for the 'full' version (e.g., title + summaries).
        model: The OpenAI embedding model to use.

    Returns:
        A tuple containing (embedding_short, embedding_full).
        Returns (None, None) if the client is not available or if API call fails.
    """
    if not openai_client:
        logging.warning("OpenAI client not initialized. Skipping embedding generation.")
        return None, None

    # Ensure inputs are strings, even if empty
    short_text = short_text or ""
    full_text = full_text or ""

    # Avoid API call if both inputs are effectively empty
    if not short_text.strip() and not full_text.strip():
        logging.debug("Skipping embedding generation: Both short and full texts are empty.")
        return None, None

    try:
        logging.debug(f"  -> Generating OpenAI embeddings using model '{model}'...")
        response = openai_client.embeddings.create(
            model=model,
            input=[short_text, full_text] # Send both texts in one request
        )
        # response.data should contain two embedding objects
        if len(response.data) == 2:
            embedding_short = response.data[0].embedding
            embedding_full = response.data[1].embedding
            logging.debug(f"  -> OpenAI embeddings generated successfully.")
            return embedding_short, embedding_full
        else:
            logging.warning(f"Unexpected number of embeddings received from OpenAI API: {len(response.data)}")
            return None, None
    except (APIError, RateLimitError) as e:
        logging.error(f"OpenAI API error during embedding generation: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error during OpenAI embedding generation: {e}", exc_info=True)
        return None, None


# --- Gemini Flash helper (yes/no classifier) -------------------------------
def is_ai_safety_post(title: str, html_body: str,
                      model="gemini-2.5-flash-preview-04-17",
                      max_chars=3500) -> bool:
    """
    Fast yes/no guard-rail: returns True iff Gemini Flash says the post's
    *primary topic* is AI-safety-related (alignment, governance, x-risk, etc.)
    """
    if not GEMINI_API_KEY:
        logging.warning("Flash guard-rail check skipped: GEMINI_API_KEY not set. Defaulting to True (fail-open).")
        return True        # fail-open so you still ingest during local tests

    # Very lightweight prompt; temp=0 for determinism
    # --- Start safe truncation ---
    snippet = markdownify(html_body or "", heading_style="ATX", bullets='-')
    snippet = snippet.encode('utf-8')[:max_chars].decode('utf-8', 'ignore')
    # --- End safe truncation ---
    prompt = f"""You are an expert AI-safety content curator.
Answer YES or NO – nothing else.

Is the following post primarily about AI safety or closely related topics (alignment, risk,
governance, technical ML safety, policy, x-risk, etc.)?

Title: {title[:120]}

Content (markdown, truncated):
{snippet}
"""
    try:
        # Consider initializing the client once globally if performance is critical
        client = genai.Client(api_key=GEMINI_API_KEY)
        rsp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        answer = (rsp.text or "").strip().upper()
        logging.debug(f"Flash guard-rail check for '{title[:50]}...': Answer={answer}")
        return "yes" in answer.lower()   # accept YES, Yes, yes, etc.
    except Exception as e:
        logging.warning(f"Flash guard-rail failed for '{title[:50]}...' ({e}); admitting post (fail-open).")
        return True  # fail-open on API hiccups

# --- Substack Fetching Helpers ---

# Helper to safely convert value to int, defaulting to 0
def safe_int_or_zero(value):
    """Safely converts value to int, defaulting to 0 if None, invalid, or TypeError."""
    if value is None: return 0
    try: return int(value)
    except (ValueError, TypeError): return 0

# Helper to parse ISO date strings reliably
def iso_to_dt(iso_string: str) -> datetime | None:
    """Converts ISO 8601 string to timezone-aware datetime object (UTC)."""
    if not iso_string:
        return None
    try:
        # Handle 'Z' suffix for UTC
        dt_obj = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        # Ensure timezone is UTC if naive
        if dt_obj.tzinfo is None:
            return dt_obj.replace(tzinfo=timezone.utc)
        # Convert to UTC if it has another timezone
        return dt_obj.astimezone(timezone.utc)
    except (ValueError, TypeError):
        logging.warning(f"Could not parse ISO date string: '{iso_string}'")
        return None

# Helper to extract Substack slug from URL or pass through if already a slug
def slug_from_thing(source: str) -> str | None:
    """
    Accepts a Substack root URL or bare 'slug' and returns the canonical host
    used in API endpoints, e.g. 'agifriday.substack.com' or 'safeai.news'.
    """
    if not source or not isinstance(source, str):
        return None

    source = source.strip()
    parsed = urlparse(source if "://" in source else f"https://{source}")

    host = parsed.netloc or parsed.path.split("/")[0]  # handles bare slugs
    if not host:
        return None

    host = host.lower().lstrip("www.")                 # normalise
    return host

# Recursive helper to extract text from body_json nodes
def extract_text(node):
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return extract_text(node.get("text") or node.get("children") or "")
    if isinstance(node, list):
        # Use newline for lists to better separate paragraphs/blocks
        return "\\n\\n".join(extract_text(c) for c in node)
    return ""

# Helper to convert Substack JSON API response to our standard post dict
def json_to_post(full: dict, slug: str) -> dict | None:
    """Converts the JSON object from Substack's post API to our standard dict format."""
    post_id = full.get("id")
    if not post_id:
        logging.warning(f"Substack post JSON missing 'id' field for slug '{slug}'. Skipping.")
        return None # ID is essential

    # Try 'published_at' first, then 'post_date', then 'updated_at'
    posted_at_dt = iso_to_dt(
        full.get("published_at")
     or full.get("post_date")
     or full.get("updated_at")
    )
    if not posted_at_dt:
        logging.warning(f"Substack post JSON missing or invalid 'published_at'/'post_date'/'updated_at' for ID {post_id}, slug '{slug}'. Skipping.")
        return None # Date is essential

    # Skip pay-walled content
    # Note: This check is done here based on the full post details
    if full.get("audience") == "only_paid" and full.get("should_show_paywall"):
        # logging.info(f"Skipping paywalled post ID {post_id} for slug '{slug}'.") # Optional info log
        return None

    # Calculate score with multiple fallbacks
    score = safe_int_or_zero(
        full.get("reaction_value") or
        full.get("public_reactions_count") or
        safe_int_or_zero(sum((full.get("reactions") or {}).values())) # Safely sum reactions
    )

    # Extract HTML body with fallbacks (including body_json and truncated_body_text)
    html_body = (
        full.get("body_html")
        or full.get("body_markdown")
        # Updated: Use extract_text helper for body_json
        or ("\\n\\n".join(extract_text(p) for p in full.get("body_json", [])))
        or f"<p>{full.get('truncated_body_text','')}</p>"
    )
    # Ensure it's never empty if description exists
    if not html_body and full.get("description"):
        html_body = f"<p>{full['description']}</p>" # Wrap description

    # Extract original tags from Substack post data (try both 'post_tags' and 'postTags')
    original_tags = []
    raw = full.get("post_tags") or full.get("postTags") or []
    for t in raw:
        # postTags items are dicts; post_tags items are strings
        if isinstance(t, dict):
            original_tags.append({"name": t.get("name")})
        elif isinstance(t, str):
            original_tags.append({"name": t})

    # --- Extract image URL (moved into json_to_post) ---
    image_url = None # Default to None
    # Try to find first image in HTML body
    if html_body:
        try:
            temp_soup = BeautifulSoup(html_body, 'html.parser')
            img_tag = temp_soup.find('img')
            if img_tag:
                # Prefer data-src if available, fallback to src
                image_url = img_tag.get('data-src') or img_tag.get('src')
        except Exception as e:
            logging.warning(f"Could not parse image from HTML body for post ID {post_id}: {e}")
    # Fallback: Use cover_image if no image found in HTML
    if not image_url:
        image_url = full.get("cover_image")
    # --- End image extraction ---

    return {
        "_id":        str(post_id), # Ensure string ID
        "title":      full.get("title", "Untitled"),
        "pageUrl":    full.get("canonical_url"),
        "postedAt":   posted_at_dt.isoformat(),
        "htmlBody":   html_body,
        "image_url":  image_url, # Add extracted image_url
        "tags":       original_tags, # Use extracted original tags
        "user":       {"displayName": slug}, # Use slug as a placeholder user/source identifier
        "coauthors":  [], # JSON API doesn't provide easily
        "baseScore":  score,
        "commentCount": full.get("comments_count", 0),
        # source_type will be added later using the mapping
    }


# ---------- Substack Iterators ----------

def iter_substack_archive(slug: str, cutoff: datetime, batch: int = 35):
    """
    Yields full post objects fetched from the Substack Archive API for a given slug,
    stopping when posts are older than the cutoff date.
    Raises exceptions on network/JSON errors.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
    }
    offset = 0
    while True:
        # slug may already be 'domain.com' or 'xyz.substack.com'
        list_url = f"https://{slug}/api/v1/archive?sort=new&search=&offset={offset}&limit={batch}"
        logging.debug(f"  Fetching archive page: slug={slug}, offset={offset}, limit={batch}")
        try:
            # Propagate errors up to the caller (fetch_substack)
            page_response = requests.get(list_url, timeout=45, headers=headers)
            page_response.raise_for_status() # Raises HTTPError for bad status codes
            # --- Handle potential list or dict response ---
            json_response = page_response.json() # Raises JSONDecodeError on invalid JSON
            if isinstance(json_response, list):
                stubs = json_response # API returned a list directly
            elif isinstance(json_response, dict):
                stubs = json_response.get("posts", []) # API returned a dict with 'posts' key
            else:
                logging.warning(f"Unexpected JSON response type ({type(json_response).__name__}) for '{slug}' at offset {offset}. Assuming empty.")
                stubs = []
            # --- End handling ---
            time.sleep(2) # Be polite
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            # Log the error origin and re-raise to signal failure
            logging.error(f"Failed fetching/decoding archive page for '{slug}' at offset {offset}: {e}")
            raise # Propagate the error

        if not stubs:
            logging.debug(f"  No more post stubs found for '{slug}' at offset {offset}.")
            break # Finished the archive for this slug

        logging.debug(f"  Processing {len(stubs)} post stubs for '{slug}'.")
        reached_cutoff = False
        for stub in stubs:
            # Check cutoff date *before* fetching full post
            posted_at_dt = iso_to_dt(stub.get("post_date"))
            if not posted_at_dt:
                logging.warning(f"Skipping stub with missing/invalid 'post_date' in archive for '{slug}': ID {stub.get('id')}")
                continue
            if posted_at_dt < cutoff:
                logging.debug(f"  Reached cutoff date ({cutoff.date()}) for '{slug}'. Stopping.")
                buffer_skip(stub.get('id'), stub.get('title'), stub.get('canonical_url'))
                reached_cutoff = True
                break # Stop processing stubs for this page

            pid = stub.get("id")
            post_title_stub = stub.get('title', 'Untitled') # Get title from stub for logging
            if not pid:
                logging.warning(f"Skipping stub with missing 'id' in archive for '{slug}'.")
                continue

            # --- Use new endpoints based on user feedback ---
            post_slug = stub.get("slug")          # e.g. "math"
            if post_slug:                         # preferred: slug endpoint
                full_post_url = f"https://{slug}/api/v1/posts/{post_slug}"
            else:                                 # fallback: new by-id endpoint
                full_post_url = f"https://{slug}/api/v1/posts/by-id/{pid}"
            # --- End endpoint update ---

            logging.info(f"    Fetching full post details for '{post_title_stub[:50]}...' (ID: {pid}) from {slug}")
            full_post_data = None # Initialize
            try:
                # Propagate errors up to the caller (fetch_substack)
                full_response = requests.get(full_post_url, timeout=45, headers=headers)
                full_response.raise_for_status()
                full_post_data = full_response.json()
                time.sleep(2) # Be polite
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                # Log the specific error but continue to next stub; fetch_substack decides overall failure
                logging.warning(f"Failed fetching/decoding full post ID {pid} for \'{slug}\': {e}. Skipping this post.")
                continue # Skip to the next stub

            # --- Check if the API returned a dict before proceeding ---
            if not isinstance(full_post_data, dict):
                logging.warning(f"Substack API for post ID {pid} ('{slug}') returned unexpected type ({type(full_post_data).__name__}) instead of dict. Skipping this post.")
                logging.debug(f"    Unexpected data was: {full_post_data}") # Log the data for debugging
                continue # Skip to the next stub
            # --- End check ---

            # Convert to standard post dict and yield if valid (handles paywall etc.)
            post_dict = json_to_post(full_post_data, slug)
            if post_dict:
                # --- Guard-rail: Check if post is AI safety related ---
                if not is_ai_safety_post(post_dict["title"], post_dict["htmlBody"]):
                    logging.info(f"    Skipping post '{post_dict['title'][:50]}...' (ID: {pid}): Failed AI safety guard-rail.")
                    buffer_skip(pid, post_dict["title"], post_dict.get("pageUrl"))
                    continue                      # discard & do NOT yield
                # Set friendly source name using the mapping
                post_dict["source_type"] = SUBSTACK_SOURCE_NAMES.get(slug, slug) # Use friendly name
                yield post_dict
            # else: # Debugging for skipped posts (paywalled, invalid date etc.)
            #     logging.debug(f"    Skipped post ID {pid} (paywalled or invalid) for '{slug}'.")

        if reached_cutoff:
            break # Stop pagination if cutoff reached

        offset += len(stubs) # Advance pagination


def iter_substack_rss(slug: str, cutoff: datetime, limit: int = 25):
    """
    Yields post objects parsed from a Substack RSS feed for a given slug,
    up to a limit and respecting the cutoff date.
    Handles feedparser errors gracefully.
    """
    # slug may already be 'domain.com' or 'xyz.substack.com'
    feed_url = f"https://{slug}/feed"
    logging.info(f"Fetching RSS feed: {feed_url}") # Keep this INFO level
    headers = { # Add headers dict
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
    }
    try:
        # Feedparser handles network errors internally to some extent
        # Pass request_headers to feedparser.parse
        d = feedparser.parse(feed_url, request_headers=headers)
        if d.bozo: # Check if feedparser encountered issues
             logging.warning(f"Error parsing RSS feed for '{slug}': {d.bozo_exception}")
             # Optionally return here if strict error handling is needed
             # return
    except Exception as e:
         # Catch unexpected errors during parsing setup
         logging.error(f"Unexpected error initializing feedparser for '{slug}': {e}")
         return # Stop iteration for this feed

    if not d or not d.entries:
        logging.info(f"RSS feed for '{slug}' is empty or could not be fetched.")
        return # Nothing to iterate

    count = 0
    for entry in d.entries:
        if count >= limit:
            break # Stop if limit reached

        # Basic sanity checks
        link = entry.get("link")
        title = entry.get("title")
        published_parsed = entry.get("published_parsed")
        if not link or not title or not published_parsed:
            logging.warning(f"Skipping RSS entry for '{slug}' due to missing link, title, or date: {link or 'No Link'}")
            continue

        # Parse date and check cutoff
        try:
            # feedparser returns struct_time, convert to datetime
            dt = datetime(*published_parsed[:6], tzinfo=timezone.utc)
        except (ValueError, TypeError):
            logging.warning(f"Skipping RSS entry for '{slug}' due to invalid date tuple: {published_parsed}")
            continue

        if dt < cutoff:
            logging.debug(f"    Skipping RSS entry (older than cutoff {cutoff.date()}): {title[:50]}...")
            buffer_skip(entry.get("id", link), title, link)
            continue # Skip posts older than cutoff

        # Extract content (prefer full content over summary)
        html_body = ""
        if entry.get("content") and isinstance(entry.get("content"), list) and len(entry.get("content")) > 0:
            html_body = entry.get("content")[0].get('value', '')
        elif entry.get("summary_detail"):
            html_body = entry.summary_detail.get('value', '')
        elif entry.get("summary"): # Fallback to basic summary
             html_body = entry.summary

        # Extract tags from RSS feed
        tag_objs = [{"name": t.term} for t in entry.tags if t and t.term] if "tags" in entry else []

        # Use link or id as fallback _id
        post_id = entry.get("id", link)

        post_dict = {
            "_id":        post_id, # Use ID or link
            "title":      title,
            "pageUrl":    link,
            "postedAt":   dt.isoformat(),
            "htmlBody":   html_body,
            "tags":       tag_objs, # Use tags parsed from RSS
            "user":       {"displayName": slug}, # Use slug as placeholder user/source
            "coauthors":  [{"displayName": entry.get("author")}] if entry.get("author") else [],
            "baseScore":  None,     # RSS has no score
            "commentCount": None,   # RSS has no comments
            "source_type": SUBSTACK_SOURCE_NAMES.get(slug, slug) # Use friendly name from mapping
        }

        # --- Guard-rail: Check if post is AI safety related ---
        if not is_ai_safety_post(post_dict["title"], post_dict["htmlBody"]):
            logging.info(f"    Skipping RSS entry '{title[:50]}...' (URL: {link}): Failed AI safety guard-rail.")
            buffer_skip(post_id, title, link)
            continue                      # discard & do NOT yield

        yield post_dict
        count += 1

# --- Substack Orchestration Function ---
def fetch_substack(source_input: str, cutoff: datetime, rss_limit: int = 25, archive_batch: int = 100, always_add_rss: bool = False):
    """
    Fetches posts for a Substack publication (given slug or URL).
    Tries the Archive API first, falls back to RSS if archive fails or returns no posts.
    Deduplicates posts by ID.

    Args:
        source_input: Substack slug or full feed/publication URL.
        cutoff: The earliest date (inclusive) for posts.
        rss_limit: Max posts to fetch via RSS fallback.
        archive_batch: Batch size for archive pagination.
        always_add_rss: If True, always fetches RSS (after archive) to catch newest posts.

    Returns:
        List of unique post dictionaries, sorted by date descending.
    """
    slug = slug_from_thing(source_input)
    if not slug:
        logging.error(f"Cannot fetch Substack: Invalid source input '{source_input}'")
        return [] # Return empty list if slug extraction fails

    posts_by_id = {}
    archive_failed = False
    archive_yielded_posts = False # Track if archive actually returned anything

    # 1. Attempt Archive Crawl
    logging.info(f"Starting Substack fetch for '{slug}' (cutoff: {cutoff.date()})")
    logging.info(f" -> Attempting Archive API crawl...")
    try:
        for post in iter_substack_archive(slug, cutoff=cutoff, batch=archive_batch):
            # iter_substack_archive already checks cutoff and AI safety guard-rail
            if post and post.get('_id'): # Ensure post and ID are valid
                posts_by_id[post['_id']] = post
                archive_yielded_posts = True # Mark that we got at least one post
        logging.info(f" -> Archive API crawl finished for '{slug}'. Found {len(posts_by_id)} posts so far.")
    except Exception as e:
        # Errors during archive crawl (network, JSON, etc.) trigger fallback
        logging.warning(f" -> Archive API crawl FAILED for '{slug}': {e}. Will attempt RSS fallback.")
        archive_failed = True

    # Check if archive finished but yielded nothing (could be new pub, 403, etc.)
    if not archive_failed and not archive_yielded_posts:
        logging.warning(f" -> Archive API crawl for '{slug}' completed but yielded no posts (older than cutoff, failed guard-rail, or other issue).")
        # Treat as failure for fallback purposes, unless always_add_rss is True
        archive_failed = True

    # 2. Decide whether to fetch RSS
    need_rss = archive_failed or always_add_rss
    if need_rss:
        reason = "fallback" if archive_failed else "always_add_rss"
        logging.info(f" -> Fetching RSS feed for '{slug}' (Reason: {reason}, Limit: {rss_limit})...")
        try:
            rss_count = 0
            for post in iter_substack_rss(slug, cutoff=cutoff, limit=rss_limit):
                # iter_substack_rss already checks cutoff and AI safety guard-rail
                 if post and post.get('_id'): # Ensure post and ID are valid
                    # Use setdefault: only add if ID wasn't already seen from archive
                    posts_by_id.setdefault(post['_id'], post)
                    rss_count +=1
            logging.info(f" -> RSS fetch finished for '{slug}'. Added/kept {rss_count} posts via RSS.")
        except Exception as e:
            # Log error but don't fail the whole process if RSS fetch fails
            logging.error(f" -> RSS fetch FAILED for '{slug}': {e}")
    else:
        logging.info(f" -> Skipping RSS fetch for '{slug}' (Archive successful and always_add_rss=False).")


    # 3. Return sorted results
    final_posts = list(posts_by_id.values())
    # Sort by 'postedAt' descending (newest first)
    final_posts.sort(key=lambda p: p.get('postedAt', ''), reverse=True)

    logging.info(f" -> Completed fetch for '{slug}'. Total unique posts: {len(final_posts)}")
    return final_posts


# --- Filtering Logic (Removed EA/LW/AF specific filters) ---
# No specific filtering logic needed here anymore, as Substack fetching
# already handles cutoff date and the AI safety guard-rail.
# Score-based filtering could be added here if desired for Substack posts.

# --- De-duplication Helper ---
def choose_highest_score(posts):
    """Keeps only the highest-scoring post for each normalized title."""
    by_title = {}
    # Filter out posts without titles first, as they cannot be deduplicated
    valid_posts = [p for p in posts if p and p.get('title')]
    original_valid_count = len(valid_posts)
    logging.info(f"\n--- Deduplicating {original_valid_count} fetched Substack posts by normalized title ---")

    removed_count = 0
    for p in valid_posts:
        # Normalize title: lowercase, replace multiple spaces with single, strip leading/trailing
        key = normalise_title(p['title'])
        # Treat None score (possible from RSS) as 0 for comparison purposes
        current_score = p.get('baseScore') or 0

        existing_entry = by_title.get(key)

        # If no entry exists for this title, add the current post
        if existing_entry is None:
            by_title[key] = p
        else:
            # If an entry exists, compare scores (treating None as 0)
            existing_score = existing_entry.get('baseScore') or 0
            # Replace if the current post has a strictly higher score
            if current_score > existing_score:
                logging.debug(f"    Replacing post '{key}' (Score {existing_score}) with higher score ({current_score})")
                by_title[key] = p
                removed_count += 1 # Count the one that was replaced
            else:
                logging.debug(f"    Keeping existing post '{key}' (Score {existing_score}), discarding lower/equal score ({current_score})")
                removed_count += 1 # Count the one being discarded

    unique_posts = list(by_title.values())
    # Calculate removed duplicates based on the difference from the initial valid count
    # duplicates_removed = original_valid_count - len(unique_posts) # Old calculation was slightly off

    logging.info(f"--- Kept {len(unique_posts)} unique posts (removed {removed_count} lower-scoring or duplicate title posts) ---")
    return unique_posts

# --- Helper for title normalization ---
def normalise_title(t: str) -> str:
    """Normalize title: lowercase, replace multiple spaces with single, strip leading/trailing."""
    if not isinstance(t, str): return "" # Handle non-string input
    return re.sub(r'\s+', ' ', t).strip().lower()

# --- Main Execution & Database Handling ---

def main():
    # Setup logging (already done at the top)
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Suppress overly verbose logs from underlying libraries (already done at the top)
    # logging.getLogger("google.generativeai").setLevel(logging.WARNING)
    # logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Check for Database URL first
    if not DATABASE_URL:
        logging.error("Error: DATABASE_URL environment variable not set. Cannot connect to database.")
        sys.exit(1) # Exit with error code

    # --- Fetch Substack Data ---
    logging.info("--- Starting Substack Feed Fetch ---")
    substack_posts_all = []
    # Option to always fetch RSS as a safety net for very new posts
    # Set this via env var or keep False
    ALWAYS_FETCH_RSS_NET = os.getenv("SUBSTACK_ALWAYS_FETCH_RSS", "False").lower() == "true"

    for feed_url_or_slug in SUBSTACK_FEEDS:
        # Use the fetch_substack function which handles archive/RSS fallback and guard-rail
        posts_for_feed = fetch_substack(
            source_input=feed_url_or_slug,
            cutoff=CUTOFF_DATE,
            rss_limit=40, # Max RSS posts per feed if needed
            archive_batch=20, # Archive API page size
            always_add_rss=ALWAYS_FETCH_RSS_NET
        )
        substack_posts_all.extend(posts_for_feed)
        # Detailed logging is now handled inside fetch_substack

    initial_fetched_count = len(substack_posts_all)
    logging.info(f"--- Finished Substack Fetch: {initial_fetched_count} posts fetched (before deduplication) ---")

    # --- Deduplicate in memory before processing ---
    unique_posts = choose_highest_score(substack_posts_all)
    total_unique_count = len(unique_posts)

    if not unique_posts:
        logging.info("No unique Substack posts remaining after deduplication. Exiting.")
        sys.exit(1) # Exit with error code

    # --- Process Posts and Insert into Database ---
    processed_count = 0
    affected_rows_count = 0 # Tracks total rows inserted
    failed_analysis_count = 0 # Track posts where analysis failed/skipped
    batch_data = [] # List to hold data tuples for batch insert
    total_failures = 0 # Track failures during batch insert
    skipped_in_db_count = 0 # Track skipped due to being in DB
    skipped_missing_data_count = 0 # Track skipped due to missing URL/Title
    skipped_analysis_error_count = 0 # Track skipped due to HTML/Markdown error
    total_skipped_by_record_count = 0 # Track posts recorded in skipped_posts table

    conn = None # Initialize conn outside the try block

    try:
        # Establish a single connection
        logging.info("\n--- Database Operations ---")
        logging.info("Connecting to the database...")
        conn = psycopg2.connect(DATABASE_URL)
        logging.info("Database connection successful.")
        register_vector(conn) # Register pgvector type with the connection
        logging.info("pgvector type registered with psycopg2 connection.")

        with conn.cursor() as cur:
            # --- Fetch existing titles early ---
            logging.info("Fetching existing titles from database...")
            cur.execute("SELECT title_norm FROM content")
            existing_titles = {row[0] for row in cur.fetchall()}
            logging.info(f"→ {len(existing_titles):,} titles already in DB")
            # --- End fetching existing titles ---

            # --- Fetch already skipped titles ---
            logging.info("Fetching already skipped titles from database...")
            cur.execute("SELECT title_norm FROM skipped_posts")
            already_skipped = {r[0] for r in cur.fetchall()}
            logging.info(f"→ {len(already_skipped):,} titles already in skipped_posts")
            # --- End fetching already skipped titles ---

            # --- Process skipped buffer ---
            if SKIPPED_BUFFER:
                extras.execute_values(cur,
                    "INSERT INTO skipped_posts (post_id, title_norm, source_url) VALUES %s "
                    "ON CONFLICT (title_norm) DO NOTHING",
                    list(SKIPPED_BUFFER))
                total_skipped_by_record_count += len(SKIPPED_BUFFER) # Add buffer size to counter
                logging.info("Inserted %s cutoff/guard-rail skips.", len(SKIPPED_BUFFER))
                # Update already_skipped cache with titles from buffer
                already_skipped.update(t for _, t, _ in SKIPPED_BUFFER)
                # Clear the buffer after successful insertion
                SKIPPED_BUFFER.clear()

            logging.info(f"\n--- Processing {total_unique_count} Unique Posts ---")
            posts_to_insert = 0 # Count posts added to batch_data
            for i, post in enumerate(unique_posts): # Iterate over unique posts
                processed_count += 1
                analysis_successful = True # Flag to track if all analyses succeed for this post

                # Safely get values, providing defaults
                url = post.get('pageUrl', 'N/A')
                title = post.get('title', 'N/A') # Get the raw title
                post_id = post.get('_id', 'Unknown ID')

                logging.info(f"\n[{i+1}/{total_unique_count}] Processing Post: '{title[:60]}...' ({url})")

                # --- Skip if URL is invalid or title missing (early checks) ---
                title_norm = normalise_title(title) # Normalize title once for checks
                if url == 'N/A' or title == 'N/A':
                     logging.warning(f"  -> Skipping post (ID: {post_id}): Missing URL or Title.")
                     record_skip(cur, post_id, title_norm, url) # Record skip
                     already_skipped.add(title_norm) # Add to in-memory cache
                     total_skipped_by_record_count += 1 # Increment counter
                     skipped_missing_data_count += 1
                     continue # Skip to the next post

                # --- Normalize title and check against existing DB titles (EARLY OUT) ---
                if title_norm in existing_titles:
                    logging.info(f"  -> Skipping post (already in DB): '{title[:60]}...'")
                    skipped_in_db_count += 1
                    continue # Already in DB, skip all further processing for this post
                # --- End early-out check ---

                # --- Check against already_skipped titles (EARLY OUT) ---
                if title_norm in already_skipped:
                    logging.info("  → Skipping (known bad): %s", title[:60])
                    # No counter increment here as it's already been recorded and counted in this run or previous
                    continue
                # --- End already_skipped check ---

                # --- Check for invalid publication date before extensive processing ---
                published_date_str = post.get('postedAt', '')
                published_date_dt = None
                if published_date_str:
                    try:
                        published_date_dt = datetime.fromisoformat(published_date_str)
                        # Check against CUTOFF_DATE (though iterators should handle this, good for safety)
                        if published_date_dt < CUTOFF_DATE:
                            logging.warning(f"  -> Skipping post (ID: {post_id}): Publication date {published_date_dt.date()} is before cutoff {CUTOFF_DATE.date()}.")
                            record_skip(cur, post_id, title_norm, url)
                            already_skipped.add(title_norm)
                            total_skipped_by_record_count += 1
                            continue
                    except ValueError:
                        logging.warning(f"  -> Skipping post (ID: {post_id}): Invalid publication date string '{published_date_str}'.")
                        record_skip(cur, post_id, title_norm, url)
                        already_skipped.add(title_norm)
                        total_skipped_by_record_count += 1
                        continue
                else: # No 'postedAt' field
                    logging.warning(f"  -> Skipping post (ID: {post_id}): Missing publication date ('postedAt').")
                    record_skip(cur, post_id, title_norm, url)
                    already_skipped.add(title_norm)
                    total_skipped_by_record_count += 1
                    continue
                # --- End publication date check ---

                # --- Extract original tags for use in cluster analysis ---
                tags_list = post.get('tags', []) # These are the original tags from Substack
                tag_names = [tag.get('name', 'N/A') for tag in tags_list if tag and isinstance(tag.get('name'), str)]

                source_type = post.get('source_type', 'Unknown') # Get added source_type

                score = post.get('baseScore') # Keep as number (or None) for DB
                comment_count = post.get('commentCount') # Keep as number (or None) for DB
                full_content = post.get('htmlBody', '') or "" # Ensure it's a string

                # --- Prepend title to HTML content ---
                if title != 'N/A' and full_content: # Only add if title exists and content exists
                    full_content = f"<h1>{title}</h1>\n\n{full_content}" # Use \n\n for markdown friendliness later
                elif title != 'N/A': # If content is empty but title exists
                    full_content = f"<h1>{title}</h1>"
                # --- End prepend title ---

                # --- Clean HTML with BeautifulSoup (Moved after duplicate check) ---
                cleaned_html = ""
                if full_content:
                    try:
                        soup = BeautifulSoup(full_content, 'html.parser')
                        for element in soup(["script", "style", "iframe", "form", "button", "input"]): # Remove unwanted tags
                            element.decompose()
                        # Optional: Add more specific cleaning here if needed
                        cleaned_html = str(soup)
                    except Exception as e:
                        logging.error(f"ERROR: BeautifulSoup cleaning failed for post '{title}' ({url}). Error: {e}. Content and analysis will be skipped.")
                        cleaned_html = "" # Keep it empty on error
                        analysis_successful = False # Skip analysis if cleaning fails
                        record_skip(cur, post_id, title_norm, url) # Record skip
                        already_skipped.add(title_norm) # Add to in-memory cache
                        total_skipped_by_record_count += 1 # Increment counter
                        skipped_analysis_error_count += 1 # Count this specific skip reason
                else:
                    logging.debug(f"  -> No HTML content found for post '{title[:60]}...', skipping cleaning.") # Info if content was empty
                    # This case implies full_content was empty. If this is a reason to skip, record it.
                    # Assuming empty HTML means we should skip and record.
                    logging.warning(f"  -> Skipping post (ID: {post_id}): HTML content is empty.")
                    record_skip(cur, post_id, title_norm, url)
                    already_skipped.add(title_norm)
                    total_skipped_by_record_count += 1
                    skipped_analysis_error_count += 1 # Or a more specific counter for empty content
                    analysis_successful = False # Mark as false if no content to analyze
                    # Don't increment skipped_count here, might still insert basic info

                # --- Convert Cleaned HTML to Markdown ---
                full_content_markdown = ""
                # Only proceed if HTML cleaning was successful (or if original content was empty but cleaning wasn't skipped)
                if analysis_successful and cleaned_html:
                    try:
                        # Use the cleaned HTML for markdown conversion
                        logging.debug(f"  -> Converting cleaned HTML to Markdown...")
                        full_content_markdown = markdownify(cleaned_html, heading_style="ATX", bullets="-")
                        logging.debug(f"  -> Markdown conversion successful.")
                    except Exception as e:
                        logging.error(f"ERROR: Could not convert cleaned HTML to Markdown for post '{title}' ({url}). Error: {e}. Content and analysis will be skipped.")
                        full_content_markdown = "" # Keep it empty on error
                        analysis_successful = False # Skip analysis if markdown fails
                        # Only count skip if cleaning didn't already fail
                        if cleaned_html is not None:
                            record_skip(cur, post_id, title_norm, url) # Record skip
                            already_skipped.add(title_norm) # Add to in-memory cache
                            total_skipped_by_record_count += 1 # Increment counter
                            skipped_analysis_error_count += 1
                elif not full_content and analysis_successful: # Handle case where original content was empty
                    logging.debug(f"  -> Original content was empty, Markdown is also empty.")
                    # This implies cleaned_html was also empty or not processed.
                    # If markdown is empty and it's a skip condition, it should have been caught by HTML check.
                    # However, if cleaning succeeded but produced empty markdown from non-empty HTML (unlikely with markdownify),
                    # or if we want to explicitly skip if full_content_markdown is empty after processing.
                    if not full_content_markdown and analysis_successful: # analysis_successful means HTML was processed
                        logging.warning(f"  -> Skipping post (ID: {post_id}): Markdown content is empty after conversion.")
                        record_skip(cur, post_id, title_norm, url)
                        already_skipped.add(title_norm)
                        total_skipped_by_record_count += 1
                        skipped_analysis_error_count += 1
                        analysis_successful = False

                else:
                     # Handle cases where markdown was empty or conversion failed earlier
                     if url != 'N/A' and skipped_analysis_error_count == 0 and skipped_missing_data_count == i + 1 - skipped_in_db_count: # Only log if not already counted as skipped
                         logging.warning(f"  -> Skipping analysis (empty/failed content): '{title[:60]}...'")
                     # analysis_successful should already be False here
                     # If analysis_successful is false at this point due to content issues,
                     # and it hasn't been recorded yet, record it.
                     if not analysis_successful and title_norm not in already_skipped:
                         logging.warning(f"  -> Recording skip for (ID: {post_id}) due to prior content processing failure.")
                         record_skip(cur, post_id, title_norm, url)
                         already_skipped.add(title_norm)
                         total_skipped_by_record_count += 1
                         # Note: skipped_analysis_error_count might have already been incremented.
                         # Avoid double counting for the print summary if an earlier specific error was caught.

                if not analysis_successful and skipped_analysis_error_count == 0 and skipped_missing_data_count == i + 1 - skipped_in_db_count: # Only count if not already skipped
                     failed_analysis_count += 1 # Increment if any analysis failed/skipped for this post

                # --- If analysis_successful is false at this point, we should not proceed to DB insertion logic.
                # The existing logic for batch_data.append should be conditional on analysis_successful OR
                # if we intend to insert posts with only basic info even if analysis failed.
                # The user request is to record skip for "empty or unparsable HTML / Markdown",
                # which implies these posts are not inserted.
                # Let's add a continue if analysis_successful is false after all content checks.

                if not analysis_successful:
                    logging.warning(f"  -> Final check: Skipping DB preparation for (ID: {post_id}) as analysis_successful is false.")
                    # Ensure it was recorded if not already. This is a safeguard.
                    if title_norm not in already_skipped:
                        record_skip(cur, post_id, title_norm, url)
                        already_skipped.add(title_norm)
                        total_skipped_by_record_count += 1
                    continue # Skip to the next post if analysis failed

                # --- AI Analysis Block ---
                logging.info(f"  -> Starting AI analysis for '{title[:60]}...'")
                sentence_summary = None
                paragraph_summary = None
                key_implication = None
                db_cluster = None
                db_tags = [] # Default for ARRAY type
                embedding_short_vector = None
                embedding_full_vector = None
                current_post_analysis_failed = False # Flag for this post's AI analysis

                # Only perform AI analysis if markdown content is available and usable
                if full_content_markdown and not full_content_markdown.isspace():
                    logging.debug(f"  Performing Gemini analysis on markdown content...")

                    sentence_summary = summarize_text(full_content_markdown)
                    if sentence_summary.startswith("Error:") or sentence_summary.startswith("Content was empty.") or sentence_summary.startswith("Analysis skipped"):
                        logging.warning(f"    Failed to generate sentence summary: {sentence_summary}")
                        current_post_analysis_failed = True
                        sentence_summary = None # Ensure it's None if failed

                    paragraph_summary = generate_paragraph_summary(full_content_markdown)
                    if paragraph_summary.startswith("Error:") or paragraph_summary.startswith("Content was empty.") or paragraph_summary.startswith("Analysis skipped"):
                        logging.warning(f"    Failed to generate paragraph summary: {paragraph_summary}")
                        current_post_analysis_failed = True
                        paragraph_summary = None # Ensure it's None if failed

                    key_implication = generate_key_implication(full_content_markdown)
                    if key_implication.startswith("Error:") or key_implication.startswith("Content was empty.") or key_implication.startswith("Analysis skipped"):
                        logging.warning(f"    Failed to generate key implication: {key_implication}")
                        current_post_analysis_failed = True
                        key_implication = None # Ensure it's None if failed

                    # Cluster Tagging
                    logging.debug(f"  Performing cluster tagging...")
                    # title and tag_names are defined earlier in the loop
                    cluster_info = generate_cluster_tag(title, tag_names, full_content_markdown) # Renamed variable
                    if cluster_info.get("error"): # Use renamed variable
                        logging.warning(f"    Failed to generate cluster/tags: {cluster_info.get('error')}") # Use renamed variable
                        current_post_analysis_failed = True
                        # db_cluster remains None, db_tags remains [] (due to initialization)
                    else:
                        db_cluster = cluster_info.get("cluster") # Use renamed variable
                        db_tags = cluster_info.get("tags")     # Use user's specified assignment (no default list)
                        if not db_cluster: # Consider it a partial failure if cluster is missing
                            logging.warning(f"    Cluster tagging returned no cluster. Tags: {db_tags}")
                            # Optionally: current_post_analysis_failed = True

                    # OpenAI Embeddings
                    # Prepare text for embeddings
                    text_for_short_embedding = title if title and title != 'N/A' else ""
                    
                    full_embedding_components = [text_for_short_embedding]
                    # Add summaries to embedding text only if they are valid strings
                    if sentence_summary and not (sentence_summary.startswith("Error:") or sentence_summary.startswith("Content was empty.") or sentence_summary.startswith("Analysis skipped")):
                        full_embedding_components.append(sentence_summary)
                    if paragraph_summary and not (paragraph_summary.startswith("Error:") or paragraph_summary.startswith("Content was empty.") or paragraph_summary.startswith("Analysis skipped")):
                        full_embedding_components.append(paragraph_summary)
                    
                    text_for_full_embedding = "\\n\\n".join(c for c in full_embedding_components if c and c.strip()).strip()

                    # Check if openai_client is available and there's text to embed
                    if openai_client and (text_for_short_embedding.strip() or text_for_full_embedding.strip()):
                        logging.debug(f"  Performing OpenAI embedding generation...")
                        embedding_short_vector, embedding_full_vector = generate_embeddings(
                            openai_client,
                            text_for_short_embedding,
                            text_for_full_embedding
                        )
                        # generate_embeddings logs its own errors and returns (None, None) on failure
                        if embedding_short_vector is None and embedding_full_vector is None:
                            logging.warning(f"    OpenAI embedding generation resulted in None for both short and full embeddings.")
                            # This is already logged by generate_embeddings, but we can note it here.
                            # Depending on strictness, could set current_post_analysis_failed = True
                    elif not openai_client:
                        logging.warning("    Skipping OpenAI embeddings: OpenAI client not initialized or not configured.")
                    else: # Text for embedding was empty
                        logging.debug("    Skipping OpenAI embeddings: Text for embedding is empty after preparation.")
                        # This case might not be a "failure" but rather a lack of input.

                else: # full_content_markdown was empty or whitespace
                    logging.warning(f"  -> Skipping AI analysis for '{title[:60]}...' because markdown content is empty or invalid.")
                    current_post_analysis_failed = True # This counts as an analysis failure if content wasn't suitable

                if current_post_analysis_failed:
                    failed_analysis_count += 1
                
                logging.info(f"  -> AI analysis finished for '{title[:60]}...'. Summary: {'Yes' if sentence_summary else 'No'}, Cluster: {'Yes' if db_cluster else 'No'}, Embeddings: {'Yes' if embedding_short_vector else 'No'}")
                # --- End AI Analysis Block ---

                # --- Extract other data ---
                # Image URL is now extracted directly in json_to_post
                image_url = post.get('image_url') # Get image_url added by json_to_post

                # Extract authors (Substack often just has the publication name/slug)
                post_authors_set = set()
                # Use 'user' displayName (slug) as primary author placeholder
                if post.get('user') and post['user'].get('displayName'):
                    post_authors_set.add(post['user']['displayName'])
                # Add coauthors if present (less common in Substack data structure)
                if post.get('coauthors'):
                    for author in post['coauthors']:
                        if author and author.get('displayName'):
                            post_authors_set.add(author['displayName'])
                authors_list = sorted(list(post_authors_set)) # Convert set to sorted list for ARRAY type
                # Ensure authors_list is never empty for the database
                if not authors_list:
                    authors_list = ['Unknown'] # Default if somehow empty

                # Extract publication date
                published_date_str = post.get('postedAt', '')
                published_date = None # Use None for DB if date is invalid
                if published_date_str:
                    try:
                        # Already timezone-aware from iso_to_dt
                        published_date = datetime.fromisoformat(published_date_str)
                    except ValueError:
                        logging.warning(f"Could not parse final datetime '{published_date_str}' for DB insertion for '{title[:50]}...'")
                        pass # Keep as None

                # AI-generated tags (db_tags) are used for the 'topics' column
                # AI-generated cluster (db_cluster) is used for the 'cluster_tag' column

                # --- Prepare data tuple for batch insertion ---
                logging.debug(f"  -> Preparing data tuple for DB insertion for post ID {post_id}.")
                # Ensure order matches DB_COLS
                data_tuple = (
                    url,                            # source_url
                    title,                          # title
                    source_type,                    # source_type
                    authors_list,                   # authors (list for ARRAY type)
                    published_date,                 # published_date (datetime or None)
                    db_tags,                        # topics (AI generated tags or None)
                    score,                          # score (int or None)
                    image_url,                      # image_url (str or None)
                    sentence_summary,               # sentence_summary (str or None)
                    paragraph_summary,              # paragraph_summary (str or None)
                    key_implication,                # key_implication (str or None)
                    cleaned_html,                   # full_content (cleaned HTML or None)
                    full_content_markdown,          # full_content_markdown (str or None)
                    comment_count,                  # comment_count (int)
                    db_cluster,                     # cluster_tag (AI generated cluster or None)
                    embedding_short_vector,         # embedding_short (list[float] or None)
                    embedding_full_vector           # embedding_full (list[float] or None)
                )
                batch_data.append(data_tuple)
                posts_to_insert += 1 # Increment count for this batch

                # --- Insert in batches ---
                if len(batch_data) >= BATCH_SIZE:
                    logging.info(f"DB Batch Insert: Attempting to insert {len(batch_data)} posts...")
                    try:
                        # Use extras.execute_batch with ON CONFLICT
                        extras.execute_batch(cur, INSERT_SQL, batch_data)
                        committed_rows = len(batch_data) # Assume all rows were potentially affected
                        conn.commit() # Commit after successful batch execution
                        logging.info(f"DB Batch Insert: Successfully committed batch ({committed_rows} rows processed).")
                        affected_rows_count += committed_rows # Use committed_rows
                    except psycopg2.DatabaseError as e:
                        logging.error(f"DB Batch Insert FAILED: {e}. Retrying {len(batch_data)} rows individually...")
                        conn.rollback() # Rollback the failed batch
                        # Retry rows individually
                        individual_successes = 0
                        for row_idx, row in enumerate(batch_data):
                            row_title = row[1] # Get title from tuple for logging
                            try:
                                # execute_values expects a list of tuples
                                extras.execute_values(cur, INSERT_SQL, [row])
                                conn.commit() # Commit each successful individual insert
                                logging.info(f"  -> DB Insert OK (Retry {row_idx+1}/{len(batch_data)}): '{row_title[:60]}...'")
                                affected_rows_count += 1
                                individual_successes += 1
                            except psycopg2.DatabaseError as inner_e:
                                logging.warning(f"  -> DB Insert FAILED (Retry {row_idx+1}/{len(batch_data)}): '{row_title[:60]}...' Error: {inner_e}. Skipping this row.")
                                conn.rollback() # Rollback the failed individual insert
                                total_failures += 1 # Count this as a failure
                        logging.info(f"DB Individual Retry completed: {individual_successes} succeeded, {len(batch_data) - individual_successes} failed.")
                    except Exception as e:
                        logging.error(f"Unexpected error during batch execution: {e}")
                        conn.rollback() # Rollback on unexpected errors
                        total_failures += len(batch_data) # Increment total failures by batch size
                        logging.error(f"Unexpected error caused failure for {len(batch_data)} rows in batch.")
                    finally:
                        batch_data = [] # Clear the batch list whether it succeeded or failed

            # --- Insert any remaining posts in the last batch ---
            if batch_data:
                logging.info(f"DB Final Batch Insert: Attempting to insert {len(batch_data)} remaining posts...")
                try:
                    extras.execute_batch(cur, INSERT_SQL, batch_data)
                    committed_rows = len(batch_data)
                    conn.commit() # Commit the final batch
                    logging.info(f"DB Final Batch Insert: Successfully committed batch ({committed_rows} rows processed).")
                    affected_rows_count += committed_rows
                except psycopg2.DatabaseError as e:
                    logging.error(f"DB Final Batch Insert FAILED: {e}. Retrying {len(batch_data)} rows individually...")
                    conn.rollback() # Rollback the failed batch
                    # Retry rows individually
                    individual_successes = 0
                    for row_idx, row in enumerate(batch_data):
                        row_title = row[1] # Get title from tuple for logging
                        try:
                            # execute_values expects a list of tuples
                            extras.execute_values(cur, INSERT_SQL, [row])
                            conn.commit() # Commit each successful individual insert
                            logging.info(f"  -> DB Insert OK (Final Retry {row_idx+1}/{len(batch_data)}): '{row_title[:60]}...'")
                            affected_rows_count += 1
                            individual_successes += 1
                        except psycopg2.DatabaseError as inner_e:
                            logging.warning(f"  -> DB Insert FAILED (Final Retry {row_idx+1}/{len(batch_data)}): '{row_title[:60]}...' Error: {inner_e}. Skipping this row.")
                            conn.rollback() # Rollback the failed individual insert
                            total_failures += 1 # Count this as a failure
                    logging.info(f"DB Final Individual Retry completed: {individual_successes} succeeded, {len(batch_data) - individual_successes} failed.")
                except Exception as e:
                    logging.error(f"Unexpected error during final batch execution: {e}")
                    conn.rollback()
                    total_failures += len(batch_data) # Increment total failures by batch size
                    logging.error(f"Unexpected error caused failure for {len(batch_data)} rows in final batch.")
                finally:
                    batch_data = [] # Clear list

    except psycopg2.OperationalError as e:
        logging.critical(f"FATAL: Database connection failed: {e}")
        sys.exit(1) # Exit with error code
    except psycopg2.DatabaseError as e:
        logging.error(f"Database error occurred: {e}")
        if conn:
            conn.rollback() # Rollback any potential changes
        sys.exit(1) # Exit with error code
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main processing loop: {e}", exc_info=True) # Log traceback
        if conn:
            conn.rollback() # Rollback on general errors too
        sys.exit(1) # Exit with error code
    finally:
        if conn:
            conn.close() # Ensure connection is closed
            logging.info("Database connection closed.")

    print(f"\n--- Processing Summary ---")
    print(f"Total Substack posts fetched (initial): {initial_fetched_count}")
    print(f"Unique posts after deduplication:       {total_unique_count}")
    print(f"Posts processed:                        {processed_count}")
    print(f"Posts skipped (missing URL/Title):      {skipped_missing_data_count}")
    print(f"Posts skipped (already in DB):          {skipped_in_db_count}")
    print(f"Posts skipped (content/analysis error): {skipped_analysis_error_count}")
    print(f"Posts recorded in skipped_posts table:  {total_skipped_by_record_count}") # Add new counter to summary
    print(f"Posts with analysis failures (Gem/Emb): {failed_analysis_count}") # Posts attempted analysis but failed
    print(f"Posts prepared for DB insertion:        {posts_to_insert}")
    print(f"DB rows affected (estimate):            {affected_rows_count}") # Estimate based on batch success/retries
    print(f"DB insert failures (individual rows):   {total_failures}")


if __name__ == "__main__":
    main()