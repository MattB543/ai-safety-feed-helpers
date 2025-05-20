#!/usr/bin/env python3
"""
Ingest top‑quality AI‑safety content from EA Forum, LessWrong, and Alignment Forum
into the `content` table of the AI‑Safety‑Feed database.

This script fetches posts via GraphQL, filters them based on date, tags, score,
and comment counts, performs analysis (summarization, implication identification,
clustering/tagging) using the Gemini API, and inserts the processed data into
a PostgreSQL database, handling potential duplicates based on normalized titles.

All Substack‑specific logic has been removed.
"""

# ================================================================
#                            Imports
# ================================================================
import os
import re
import json
import time
import logging
from datetime import datetime, timezone, date # Explicit imports for clarity
import sys # For exiting early

import requests
from markdownify import markdownify
from bs4 import BeautifulSoup
import psycopg2
from psycopg2 import extras # Explicit import for batch insertion
from pgvector.psycopg2 import register_vector # <<< ADD THIS IMPORT
from dotenv import load_dotenv
from google import genai as genai
from google.genai import types # Explicit import for types
from openai import OpenAI # Add OpenAI import
from openai import APIError, RateLimitError # Optional: for more specific error handling

# ================================================================
#                      Environment & Setup
# ================================================================
load_dotenv()  # Load .env BEFORE using env vars

# --- Essential Environment Variables ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DATABASE_URL   = os.environ.get("AI_SAFETY_FEED_DB_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # <<< ADD THIS

# --- Initial Checks ---
if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not set. Cannot perform analysis.")
    sys.exit(1) # Exit if API key is missing

if not DATABASE_URL:
    print("CRITICAL ERROR: AI_SAFETY_FEED_DB_URL environment variable not set. Cannot connect to database.")
    sys.exit(1) # Exit if DB URL is missing

if not OPENAI_API_KEY: # <<< ADD THIS CHECK
    print("CRITICAL ERROR: OPENAI_API_KEY environment variable not set. Cannot generate embeddings.")
    logging.critical("CRITICAL ERROR: OPENAI_API_KEY environment variable not set. Cannot generate embeddings.") # Also log
    sys.exit(1) # Exit if OpenAI key is missing

# --- Logging Configuration (Optional but recommended) ---
# Basic logging setup - consider more advanced config for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING) # Added for requests/urllib3 noise
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING) # Suppress OpenAI logs if needed

# ================================================================
#                     Forum‑specific constants
# ================================================================
EA_API_URL = "https://forum.effectivealtruism.org/graphql"
EA_AI_SAFETY_TAG_ID = "oNiQsBHA3i837sySD"

LW_API_URL = "https://www.lesswrong.com/graphql"
LW_AI_SAFETY_TAG_ID = "yBXKqk8wEg6eM8w5y"

AF_API_URL = "https://www.alignmentforum.org/graphql"  # AF uses 'top' view, no specific tag filter needed here

DEFAULT_LIMIT = 3000 # Max posts to fetch per source initially
BATCH_SIZE    = 3   # Rows per database INSERT batch

# Ingest only posts published on/after this date (UTC, inclusive)
CUTOFF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)

# ================================================================
#                 Filtering Thresholds & Tag Sets
# ================================================================
# --- Score/Comment Thresholds ---
# EA Forum
EA_SCORE_THRESHOLD_HIGH         = 85
EA_COMMENT_THRESHOLD_HIGH_SCORE = 0
EA_SCORE_THRESHOLD_MID          = 65
EA_COMMENT_THRESHOLD_MID_SCORE  = 12

# LessWrong
LW_SCORE_THRESHOLD_HIGH         = 85
LW_COMMENT_THRESHOLD_HIGH_SCORE = 0
LW_SCORE_THRESHOLD_MID          = 65
LW_COMMENT_THRESHOLD_MID_SCORE  = 20

# Alignment Forum
AF_SCORE_THRESHOLD_HIGH         = 140
AF_COMMENT_THRESHOLD_HIGH_SCORE = -1 # Effectively means no minimum comments for high score posts
AF_SCORE_THRESHOLD_MID          = 100
AF_COMMENT_THRESHOLD_MID_SCORE  = 20

# --- Tag Sets for Filtering ---
APRIL_FOOLS_TAGS = {"April Fool's", "April Fools' Day"} # Set for fast lookups
AI_TAGS_LW       = {"AI"} # Required tag for LW posts

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
    "embedding_short",
    "embedding_full"
)
NUM_DB_COLS = len(DB_COLS) # Calculate number of placeholders needed

# Pre-compute the INSERT SQL statement for efficiency
INSERT_SQL = f"""
INSERT INTO content ({', '.join(DB_COLS)})
VALUES ({', '.join(['%s'] * NUM_DB_COLS)})
ON CONFLICT (title_norm) DO NOTHING;
"""

# SQL and helper for recording skipped posts
SKIP_INSERT_SQL = """
INSERT INTO skipped_posts (post_id, title_norm, source_url)
VALUES (%s, %s, %s)
ON CONFLICT (title_norm) DO NOTHING;
"""

def record_skip(cur, post_id: str, title_norm: str, source_url: str | None):
    """Insert one row into skipped_posts (no commit)."""
    # Ensure post_id and source_url are strings, handle None for source_url
    cur.execute(SKIP_INSERT_SQL, (
        str(post_id) if post_id is not None else 'N/A',
        str(title_norm),
        str(source_url) if source_url is not None else None
    ))

# ================================================================
#                          Gemini Helpers
# ================================================================

def call_gemini_api(prompt: str, model_name: str = "gemini-2.5-pro-preview-03-25") -> str:
    """
    Utility wrapper around the Gemini API.

    Calls the specified Gemini model with the given prompt.
    Handles common API errors and returns the generated text content
    or a string indicating the error type.

    Args:
        prompt: The text prompt to send to the Gemini API.
        model_name: The specific Gemini model to use.

    Returns:
        The generated text content as a string, or an error message
        string starting with "Error:" or "Analysis skipped".
    """
    # API Key check is done globally at startup, but double-check here is harmless
    if not GEMINI_API_KEY:
        logging.warning("call_gemini_api called without GEMINI_API_KEY (should have been caught earlier).")
        return "Analysis skipped (missing API key)."

    logging.debug(f"Calling Gemini API (model: {model_name}) with prompt (first 100 chars): {prompt[:100]}...")
    try:
        # Instantiate the client inside the function (legacy way, as per original script)
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2), # Low temp for more deterministic output
        )

        # Extract text content robustly
        if hasattr(response, 'text'):
            result = response.text.strip()
        elif hasattr(response, 'parts') and response.parts:
            result = "".join(part.text for part in response.parts).strip()
        else:
            logging.warning(f"Gemini API response structure was unexpected: {response}")
            result = "Error: Could not extract text from API response."

        # Basic check for empty or potentially problematic results
        if not result:
             logging.warning(f"Gemini API returned an empty result for prompt: {prompt[:100]}...")
             result = "Error: Analysis returned empty result." # Provide specific error
        elif result.startswith("Error:"):
             logging.warning(f"Gemini API returned an error message: '{result}'")
             # Return the error message as is

        logging.debug(f"Gemini API call successful. Result (first 100 chars): {result[:100]}...")
        return result

    except types.generation_types.BlockedPromptException as e:
        logging.error(f"Gemini API call failed due to blocked prompt: {e}")
        print(f"ERROR: Gemini API call failed due to blocked prompt. Prompt (start): {prompt[:100]}...") # Also print for visibility
        return "Error: Analysis blocked due to prompt content."
    except types.generation_types.StopCandidateException as e:
        logging.error(f"Gemini API call failed due to stop candidate: {e}")
        print(f"ERROR: Gemini API call failed due to stop candidate. Prompt (start): {prompt[:100]}...") # Also print for visibility
        return "Error: Analysis stopped unexpectedly by the model."
    except Exception as e: # Catch other potential API errors
        logging.error(f"Unexpected error during Gemini API call: {e}", exc_info=True) # Log traceback
        print(f"ERROR: Unexpected error during Gemini API call: {e}. Prompt (start): {prompt[:100]}...") # Also print for visibility
        return f"Error during analysis: {e}"

# ---------- Specific Analysis Functions ----------

def summarize_text(text_to_summarize: str) -> str:
    """Generates a concise 1-2 sentence summary using the Gemini API."""
    if not text_to_summarize or text_to_summarize.isspace():
        logging.info("Skipping sentence summary: Input content was empty.")
        return "Content was empty."

    # Detailed prompt for better control
    prompt = f"""
Summarize the following AI safety content in 2 concise sentences (maximum 50 words).
Focus on the core argument, key insight, or main conclusion rather than methodology.
Use clear, accessible language while preserving technical accuracy.
The summary should be very readable and help readers quickly understand what makes this content valuable or interesting and decide if they want to read more.

--- Content to summarize ---
{text_to_summarize}
"""
    return call_gemini_api(prompt)

def generate_paragraph_summary(text_to_summarize: str) -> str:
    """Generates a structured paragraph summary using the Gemini API."""
    if not text_to_summarize or text_to_summarize.isspace():
        logging.info("Skipping paragraph summary: Input content was empty.")
        return "Content was empty."

    # Detailed prompt specifying structure and rules
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
{text_to_summarize}
"""
    return call_gemini_api(prompt)

def generate_key_implication(text_to_analyze: str) -> str:
    """Identifies the single most important logical consequence using the Gemini API."""
    if not text_to_analyze or text_to_analyze.isspace():
        logging.info("Skipping key implication: Input content was empty.")
        return "Content was empty."

    # Detailed prompt focusing on actionable insight
    prompt = f"""
Based on the AI safety content below, identify the single most important logical consequence or implication in one concise sentence (25-35 words). Focus on:

-   What change in thinking, strategy, or priorities follows from accepting this content's conclusions?
-   How might this alter our understanding of AI safety or governance approaches?
-   A specific actionable insight rather than a general statement of importance.
-   The "so what" that would matter to an informed AI safety community member.

The implication should represent a direct consequence of the content's argument, not simply restate the main point.

--- Content to analyze ---
{text_to_analyze}
"""
    return call_gemini_api(prompt)

def generate_cluster_tag(title: str, tags_list: list[str], content_markdown: str) -> dict:
    """
    Generates a cluster and canonical tags using the Gemini API based on provided taxonomy.

    Args:
        title: The title of the post.
        tags_list: List of original author-provided tags (can be empty).
        content_markdown: The markdown content of the post.

    Returns:
        dict: Parsed JSON like {"cluster": "...", "tags": ["..."]} on success.
        dict: {"error": "Reason string"} on failure (API error, empty content, parsing error).
    """
    if not content_markdown or content_markdown.isspace():
        logging.info("Skipping cluster/tag generation: Input content was empty.")
        return {"error": "Content was empty."}
    if not title:
        title = "Untitled" # Provide a default if title is missing
    if not tags_list:
        tags_list = ["N/A"] # Provide a default if tags are missing

    # Detailed prompt including the taxonomy
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
    raw_response = call_gemini_api(prompt)

    # Check if the API call itself returned an error string
    if raw_response.startswith("Error:") or raw_response.startswith("Analysis skipped"):
        logging.warning(f"Cluster tag generation failed at API call stage: {raw_response}")
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
        if isinstance(parsed_json, dict) and \
           "cluster" in parsed_json and isinstance(parsed_json["cluster"], str) and \
           "tags" in parsed_json and isinstance(parsed_json["tags"], list):
            # Further validation: ensure tags are strings
            if all(isinstance(tag, str) for tag in parsed_json["tags"]):
                logging.debug(f"Successfully parsed cluster/tag JSON: {parsed_json}")
                return parsed_json
            else:
                logging.warning(f"Parsed JSON from cluster tag API has non-string items in tags list: {parsed_json['tags']}")
                return {"error": "Parsed JSON tags list contains non-string items"}
        else:
            logging.warning(f"Parsed JSON from cluster tag API has unexpected structure or types: {parsed_json}")
            return {"error": "Parsed JSON has unexpected structure or types"}
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from cluster tag API response. Error: {e}. Cleaned response: '{cleaned_response}'")
        print(f"ERROR: Failed to parse JSON from cluster tag API response: {e}. Cleaned response was: '{cleaned_response}'") # Also print
        return {"error": f"Failed to parse JSON response: {e}"}
    except Exception as e: # Catch other potential errors during parsing/validation
        logging.error(f"Unexpected error processing cluster tag API response: {e}", exc_info=True)
        print(f"ERROR: Unexpected error processing cluster tag API response: {e}") # Also print
        return {"error": f"Unexpected error processing cluster tag response: {e}"}

# ================================================================
#                     OpenAI Embedding Helper
# ================================================================

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
        logging.warning("OpenAI client not initialized (should have been caught earlier). Skipping embedding generation.")
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
        # Use the globally initialized client
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
        print(f"ERROR: OpenAI API error during embedding generation: {e}") # Also print
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error during OpenAI embedding generation: {e}", exc_info=True)
        print(f"ERROR: Unexpected error during OpenAI embedding generation: {e}") # Also print
        return None, None

# ================================================================
#                         Utility Helpers
# ================================================================

def normalise_title(title: str) -> str:
    """
    Normalizes a title string for consistent comparison and database indexing.
    Converts to lowercase, replaces multiple whitespace chars with a single space,
    and strips leading/trailing whitespace.

    Args:
        title: The original title string.

    Returns:
        The normalized title string.
    """
    if not title: return ""
    return re.sub(r"\s+", " ", title).strip().lower()

def iso_to_dt(iso_string: str | None) -> datetime | None:
    """
    Safely converts an ISO 8601 timestamp string to a timezone-aware
    datetime object (UTC). Handles 'Z' notation and naive datetimes.

    Args:
        iso_string: The ISO 8601 formatted string.

    Returns:
        A timezone-aware datetime object (UTC) or None if parsing fails.
    """
    if not iso_string:
        return None
    try:
        # Handle 'Z' suffix for UTC and ensure timezone awareness
        dt_obj = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        if dt_obj.tzinfo is None:
            # Assume UTC if timezone is naive
            return dt_obj.replace(tzinfo=timezone.utc)
        # Convert to UTC if it has another timezone
        return dt_obj.astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse ISO date string: '{iso_string}'. Error: {e}")
        return None

def safe_int_or_zero(value: any) -> int:
    """
    Safely converts a value to an integer. Returns 0 if the value is None,
    cannot be converted, or causes a TypeError/ValueError.

    Args:
        value: The value to convert.

    Returns:
        The integer representation or 0.
    """
    if value is None: return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def safe_join(separator: str, items: list[str | None]) -> str:
    """
    Joins a list of strings or None with a separator, skipping None or empty/whitespace items.
    """
    return separator.join(item for item in items if item and isinstance(item, str) and item.strip())

# ================================================================
#                      GraphQL Fetching Logic
# ================================================================

# Define the fields we want to retrieve for each post
POST_FIELDS = """
  _id           # Unique identifier for the post
  title         # Post title
  pageUrl       # Canonical URL of the post
  commentCount  # Number of comments
  baseScore     # Score/karma of the post
  postedAt      # Publication timestamp (ISO 8601)
  htmlBody      # Full HTML content of the post
  tags {        # Associated tags
    _id
    name
  }
  user {        # Primary author information
    displayName
  }
  coauthors {   # Co-author information (if any)
    displayName
  }
"""

def get_forum_posts(api_url: str, tag_id: str | None = None, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """
    Queries a forum's GraphQL API for posts.

    Handles fetching posts either by a specific tag ID (for EA/LW) or
    the default 'top' view (for AF). Includes error handling and logging.

    Args:
        api_url: The GraphQL endpoint URL for the forum.
        tag_id: The specific tag ID to filter by (optional).
        limit: The maximum number of posts to retrieve.

    Returns:
        A list of post dictionaries, or an empty list if fetching fails or
        no posts are found.
    """
    # Determine the view clause based on whether a tag_id is provided
    if tag_id:
        view_clause = f'view: "tagById", tagId: "{tag_id}"'
        source_desc = f"{api_url} (Tag ID: {tag_id})"
    else:
        view_clause = 'view: "top"' # Default view for forums like AF
        source_desc = f"{api_url} (View: top)"

    # Construct the GraphQL query using an f-string
    query = f"""
    {{
      posts(
        input: {{
          terms: {{
            {view_clause}
            limit: {limit}
            # Add other terms like 'sort', 'before', 'after' if needed later
          }}
        }}
      ) {{
        results {{
          {POST_FIELDS}
        }}
      }}
    }}
    """

    # Standard headers for the request
    headers = {
        "Content-Type": "application/json",
        # Use a more descriptive User-Agent
        "User-Agent": "AI-Safety-Feed-Ingestion-Script/1.0 (https://github.com/your-repo; contact@example.com)"
    }

    print(f"Executing GraphQL query for {limit} posts from {source_desc}...")
    logging.info(f"Executing GraphQL query for {limit} posts from {source_desc}")

    try:
        response = requests.post(api_url, json={"query": query}, headers=headers, timeout=90) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        result = response.json()

        # Check for GraphQL-specific errors returned in the response body
        if "errors" in result:
            logging.error(f"GraphQL API ({source_desc}) returned errors: {json.dumps(result['errors'], indent=2)}")
            print(f"ERROR: GraphQL API ({source_desc}) returned errors. Check logs.")
            # Optionally log the failed query for debugging (be careful with sensitive data if any)
            # logging.debug(f"Failed GraphQL query was:\n{query}")
            return [] # Return empty list on GraphQL errors

        # Check for expected data structure
        if "data" not in result or "posts" not in result["data"] or "results" not in result["data"]["posts"]:
             logging.warning(f"Unexpected response structure from {source_desc}. 'data.posts.results' not found.")
             print(f"WARNING: Unexpected response structure from {source_desc}. Check logs.")
             # Log the actual data received for debugging
             # logging.debug(f"Response data received: {json.dumps(result.get('data', {}), indent=2)}")
             return [] # Return empty list if structure is wrong

        posts_data = result["data"]["posts"]["results"]
        print(f"Successfully fetched {len(posts_data)} posts from {source_desc}.")
        logging.info(f"Successfully fetched {len(posts_data)} posts from {source_desc}.")
        return posts_data

    except requests.exceptions.Timeout:
        logging.error(f"Query failed for {source_desc}: Request timed out.")
        print(f"ERROR: Query failed for {source_desc}: Request timed out.")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Query failed for {source_desc}: {e}", exc_info=True)
        print(f"ERROR: Query failed for {source_desc}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response status code: {e.response.status_code}")
            # Avoid logging potentially large response bodies directly unless debugging
            # logging.debug(f"Response body (truncated): {e.response.text[:500]}...")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response from {source_desc}: {e}")
        print(f"ERROR: Failed to decode JSON response from {source_desc}. Check logs.")
        # Log the response text that failed to parse
        # if hasattr(response, 'text'):
        #     logging.debug(f"Response text was (truncated): {response.text[:500]}...")
        return []
    except Exception as e:
        # Catch any other unexpected errors during the fetch process
        logging.error(f"An unexpected error occurred fetching posts from {source_desc}: {e}", exc_info=True)
        print(f"ERROR: An unexpected error occurred fetching posts from {source_desc}. Check logs.")
        return []

# ================================================================
#                     Source‑Specific Filtering
# ================================================================

def filter_ea_posts(posts: list[dict], tag_id: str) -> list[dict]:
    """Filters EA Forum posts based on date, AI safety tag, score/comments, and excludes April Fools'."""
    print(f"\n--- Filtering {len(posts)} EA Forum posts ---")
    if not posts: return []

    filtered_posts = []
    for p in posts:
        # Basic check for essential fields
        if not p or not p.get('pageUrl') or not p.get('postedAt'):
            logging.debug(f"Skipping EA post due to missing essential fields: {p.get('_id', 'N/A')}")
            continue

        # 1. Filter by date
        posted_at_dt = iso_to_dt(p.get("postedAt"))
        if not posted_at_dt or posted_at_dt < CUTOFF_DATE:
            continue # Skip if date is invalid or before cutoff

        # 2. Check tags: Must have AI safety tag, must NOT have April Fools' tag
        post_tags = p.get("tags", [])
        has_ai_safety_tag = any(t and t.get("_id") == tag_id for t in post_tags)
        has_april_fools_tag = any(t and t.get("name") in APRIL_FOOLS_TAGS for t in post_tags)

        if not has_ai_safety_tag or has_april_fools_tag:
            continue # Skip if wrong tags

        # 3. Apply score and comment filter using defined thresholds
        score = safe_int_or_zero(p.get("baseScore"))
        comments = safe_int_or_zero(p.get("commentCount"))

        passes_threshold = (
            (score >= EA_SCORE_THRESHOLD_HIGH and comments >= EA_COMMENT_THRESHOLD_HIGH_SCORE) or
            (score > EA_SCORE_THRESHOLD_MID and comments > EA_COMMENT_THRESHOLD_MID_SCORE)
        )

        if passes_threshold:
            p["source_type"] = "EA Forum" # Add source type identifier
            filtered_posts.append(p)

    print(f"--- Found {len(filtered_posts)} EA Forum posts meeting criteria ---")
    logging.info(f"Filtered EA Forum posts: {len(posts)} -> {len(filtered_posts)}")
    return filtered_posts

def filter_lw_posts(posts: list[dict]) -> list[dict]:
    """Filters LessWrong posts based on date, 'AI' tag, score/comments, and excludes April Fools'."""
    print(f"\n--- Filtering {len(posts)} LessWrong posts ---")
    if not posts: return []

    filtered_posts = []
    for p in posts:
        # Basic check
        if not p or not p.get('pageUrl') or not p.get('postedAt'):
            logging.debug(f"Skipping LW post due to missing essential fields: {p.get('_id', 'N/A')}")
            continue

        # 1. Filter by date
        posted_at_dt = iso_to_dt(p.get("postedAt"))
        if not posted_at_dt or posted_at_dt < CUTOFF_DATE:
            continue

        # 2. Check tags: Must include "AI", must NOT include April Fools'
        # Use set for efficient lookup
        tag_names = {t.get("name") for t in p.get("tags", []) if t and t.get("name")}
        has_ai_tag = bool(AI_TAGS_LW & tag_names) # Check intersection with required AI tags
        has_april_fools_tag = bool(APRIL_FOOLS_TAGS & tag_names) # Check intersection with April Fools tags

        if not has_ai_tag or has_april_fools_tag:
            continue

        # 3. Apply score and comment filter
        score = safe_int_or_zero(p.get("baseScore"))
        comments = safe_int_or_zero(p.get("commentCount"))

        passes_threshold = (
            (score >= LW_SCORE_THRESHOLD_HIGH and comments >= LW_COMMENT_THRESHOLD_HIGH_SCORE) or
            (score > LW_SCORE_THRESHOLD_MID and comments > LW_COMMENT_THRESHOLD_MID_SCORE)
        )

        if passes_threshold:
            p["source_type"] = "Less Wrong" # Add source type
            filtered_posts.append(p)

    print(f"--- Found {len(filtered_posts)} LessWrong posts meeting criteria ---")
    logging.info(f"Filtered LessWrong posts: {len(posts)} -> {len(filtered_posts)}")
    return filtered_posts

def filter_af_posts(posts: list[dict]) -> list[dict]:
    """Filters Alignment Forum posts based on date, score/comments, and excludes April Fools'."""
    print(f"\n--- Filtering {len(posts)} Alignment Forum posts ---")
    if not posts: return []

    filtered_posts = []
    for p in posts:
        # Basic check
        if not p or not p.get('pageUrl') or not p.get('postedAt'):
            logging.debug(f"Skipping AF post due to missing essential fields: {p.get('_id', 'N/A')}")
            continue

        # 1. Filter by date
        posted_at_dt = iso_to_dt(p.get("postedAt"))
        if not posted_at_dt or posted_at_dt < CUTOFF_DATE:
            continue

        # 2. Check tags: Must NOT include April Fools'
        # AF doesn't require a specific positive tag like EA/LW for this script's purpose
        tag_names = {t.get("name") for t in p.get("tags", []) if t and t.get("name")}
        has_april_fools_tag = bool(APRIL_FOOLS_TAGS & tag_names)

        if has_april_fools_tag:
            continue

        # 3. Apply score and comment filter
        score = safe_int_or_zero(p.get("baseScore"))
        comments = safe_int_or_zero(p.get("commentCount"))

        # Note the different comment threshold logic for AF High score
        passes_threshold = (
            (score >= AF_SCORE_THRESHOLD_HIGH and comments > AF_COMMENT_THRESHOLD_HIGH_SCORE) or # '>' check for comments
            (score > AF_SCORE_THRESHOLD_MID and comments > AF_COMMENT_THRESHOLD_MID_SCORE)
        )

        if passes_threshold:
            p["source_type"] = "Alignment Forum" # Add source type
            filtered_posts.append(p)

    print(f"--- Found {len(filtered_posts)} Alignment Forum posts meeting criteria ---")
    logging.info(f"Filtered Alignment Forum posts: {len(posts)} -> {len(filtered_posts)}")
    return filtered_posts

# ================================================================
#                     Deduplication Helper
# ================================================================

def choose_highest_score(posts: list[dict]) -> list[dict]:
    """
    Deduplicates a list of post dictionaries based on normalized titles.
    If multiple posts share the same normalized title, only the one with
    the highest score is kept. Posts without titles are discarded.

    Args:
        posts: A list of post dictionaries.

    Returns:
        A list of unique post dictionaries, keeping the highest-scoring duplicates.
    """
    print(f"\n--- Deduplicating {len(posts)} posts in memory by normalized title (keeping highest score) ---")
    posts_by_norm_title: dict[str, dict] = {}
    valid_post_count = 0
    discarded_no_title = 0

    for p in posts:
        title = p.get("title")
        if not title:
            discarded_no_title += 1
            continue # Cannot deduplicate without a title

        valid_post_count += 1
        norm_title = normalise_title(title)
        current_score = safe_int_or_zero(p.get("baseScore"))

        # Check if we've seen this normalized title before
        existing_post = posts_by_norm_title.get(norm_title)

        if existing_post is None:
            # First time seeing this title, add it
            posts_by_norm_title[norm_title] = p
        else:
            # Duplicate title found, compare scores
            existing_score = safe_int_or_zero(existing_post.get("baseScore"))
            if current_score > existing_score:
                # Current post has a higher score, replace the existing one
                posts_by_norm_title[norm_title] = p
            # Otherwise, keep the existing higher-scoring post

    unique_posts = list(posts_by_norm_title.values())
    duplicates_removed = valid_post_count - len(unique_posts)

    if discarded_no_title > 0:
        print(f"--- Discarded {discarded_no_title} posts lacking a title during deduplication ---")
    print(f"--- Kept {len(unique_posts)} unique posts (removed {duplicates_removed} lower-scoring duplicates) ---")
    logging.info(f"Deduplication: Input {len(posts)}, Valid w/ Title {valid_post_count}, Unique Output {len(unique_posts)}")
    return unique_posts

# ================================================================
#                     Main Processing Logic
# ================================================================

def main():
    """
    Main execution function:
    1. Fetches posts from EA, LW, AF.
    2. Filters posts based on criteria.
    3. Deduplicates posts by title, keeping highest score.
    4. Connects to the database.
    5. Fetches existing titles to avoid reprocessing.
    6. Processes each unique post: cleans HTML, converts to Markdown, runs Gemini analyses.
    7. Inserts processed data into the database in batches.
    8. Prints summary statistics.
    """
    start_time = time.time()
    print("================================================================")
    print(f"Starting AI Safety Feed Ingestion Script at {datetime.now(timezone.utc)}")
    print("================================================================")

    # -------- 1. Fetch Raw Data --------
    print("\n--- Fetching Raw Posts ---")
    ea_raw = get_forum_posts(EA_API_URL, EA_AI_SAFETY_TAG_ID, limit=DEFAULT_LIMIT)
    lw_raw = get_forum_posts(LW_API_URL, LW_AI_SAFETY_TAG_ID, limit=DEFAULT_LIMIT)
    af_raw = get_forum_posts(AF_API_URL, limit=DEFAULT_LIMIT) # No tag ID for AF

    # -------- 2. Filter Data --------
    print("\n--- Filtering Posts ---")
    ea_posts = filter_ea_posts(ea_raw, EA_AI_SAFETY_TAG_ID)
    lw_posts = filter_lw_posts(lw_raw)
    af_posts = filter_af_posts(af_raw)

    # -------- 3. Combine & Deduplicate --------
    combined_filtered_posts = ea_posts + lw_posts + af_posts
    initial_filtered_count = len(combined_filtered_posts)
    print(f"\n--- Total posts from all sources after initial filtering: {initial_filtered_count} ---")

    unique_posts = choose_highest_score(combined_filtered_posts)
    total_unique_count = len(unique_posts)

    if not unique_posts:
        print("\nNo unique posts remaining after filtering and deduplication. Nothing to process. Exiting.")
        logging.info("No unique posts remaining after filtering and deduplication. Exiting.")
        return # Exit gracefully if no posts left

    # -------- 4. Database Connection & Setup --------
    conn = None
    processed_count = 0
    affected_rows_count = 0 # Tracks total rows successfully inserted/updated
    failed_analysis_count = 0 # Track posts where *any* analysis step failed/skipped
    batch_data = [] # List to hold data tuples for batch insert
    total_db_failures = 0 # Track rows in failed batches
    embedding_failures_count = 0 # Track embedding generation failures
    total_skipped_recorded_in_db_count = 0 # New counter for skips recorded in DB

    try:
        print("\n--- Connecting to Database ---")
        logging.info(f"Connecting to database using URL: {DATABASE_URL[:20]}...") # Log partial URL
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False # Ensure transactions are handled manually
        register_vector(conn) # <<< REGISTER VECTOR TYPE HANDLER
        print("Database connection successful.")
        logging.info("Database connection successful.")

        # Initialize OpenAI client here
        print("Initializing OpenAI client...")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logging.info("OpenAI client initialized.")

        # -------- 5. Fetch Existing Titles --------
        with conn.cursor() as cur: # Use 'with' for automatic cursor closing
            print("Fetching existing normalized titles from database...")
            cur.execute("SELECT title_norm FROM content")
            # Fetchall can be memory intensive for huge tables, but likely okay here.
            # Consider server-side cursors or LIMIT/OFFSET for very large tables.
            existing_titles = {row[0] for row in cur.fetchall()}
            print(f"--> Found {len(existing_titles):,} existing titles in the database.")
            logging.info(f"Fetched {len(existing_titles)} existing titles.")

            # Fetch already skipped titles
            print("Fetching already skipped normalized titles from database...")
            cur.execute("SELECT title_norm FROM skipped_posts")
            already_skipped = {r[0] for r in cur.fetchall()}
            print(f"--> Found {len(already_skipped):,} already skipped titles in the database.")
            logging.info(f"Fetched {len(already_skipped)} already skipped titles.")

            # -------- 6. Process Posts & Perform Analysis --------
            print(f"\n--- Starting Processing and Analysis for {total_unique_count} Unique Posts ---")
            for i, post in enumerate(unique_posts):
                processed_count += 1
                post_id = post.get('_id', 'N/A') # For logging
                title = post.get('title', 'Untitled')
                url = post.get('pageUrl', 'N/A')
                print(f"\n[{processed_count}/{total_unique_count}] Processing Post: '{title[:70]}...' (ID: {post_id})")
                logging.info(f"Processing post {processed_count}/{total_unique_count}: ID {post_id}, Title: {title[:70]}...")

                # --- 6a. Early Skip: Check if already in DB or marked as skipped ---
                norm_title = normalise_title(title)
                if norm_title in already_skipped: # Check this first
                    print(f"  -> Skipping (already marked as skipped or failed this run): Normalized title '{norm_title}'.")
                    logging.info(f"  Skipping post ID {post_id} (Title: {title[:70]}...) - already marked as skipped (in skipped_posts or this run).")
                    continue # Skip to the next post
                if norm_title in existing_titles:
                    print(f"  -> Skipping (already in content table): Normalized title '{norm_title}'.")
                    logging.info(f"  Skipping post ID {post_id} (Title: {title[:70]}...) - already in content table.")
                    continue # Skip to the next post

                # --- 6b. Initialize Analysis Variables ---
                analysis_successful_flag = True # Assume success initially for this post
                sentence_summary = None
                paragraph_summary = None
                key_implication = None
                db_cluster = None   # Store the extracted cluster string for DB
                db_tags = None      # Store the extracted tags list for DB
                full_content_markdown = None # Initialize markdown content
                embedding_short_vector = None # Initialize short embedding
                embedding_full_vector = None  # Initialize full embedding

                # --- 6c. Extract & Clean Content ---
                print("  -> Cleaning HTML and converting to Markdown...")
                html_body = post.get('htmlBody', '') or "" # Ensure it's a string

                # Prepend title for context (optional, but can help analysis)
                if title != 'Untitled' and html_body:
                    html_body = f"<h1>{title}</h1>\n\n{html_body}"
                elif title != 'Untitled':
                    html_body = f"<h1>{title}</h1>"

                cleaned_html = ""
                if html_body:
                    try:
                        soup = BeautifulSoup(html_body, 'html.parser')
                        # Remove script/style tags which interfere with markdown/analysis
                        for element in soup(["script", "style", "noscript"]):
                            element.decompose()
                        # Optional: Add more cleaning steps here (e.g., remove ads, specific divs)
                        cleaned_html = str(soup)
                        print("  -> HTML cleaning successful.")
                    except Exception as e:
                        logging.error(f"BeautifulSoup cleaning failed for post ID {post_id} ('{title[:50]}...'). Error: {e}", exc_info=True)
                        print(f"  ERROR: HTML cleaning failed: {e}. Marking for skip and continuing.")
                        record_skip(cur, post_id, norm_title, url)
                        conn.commit() # Commit the skip record
                        already_skipped.add(norm_title)
                        total_skipped_recorded_in_db_count += 1
                        continue # Skip this post
                else:
                    print("  -> Post has no HTML body content. Marking for skip and continuing.")
                    logging.warning(f"Post ID {post_id} ('{title[:50]}...') has no HTML body. Marking for skip.")
                    record_skip(cur, post_id, norm_title, url)
                    conn.commit() # Commit the skip record
                    already_skipped.add(norm_title)
                    total_skipped_recorded_in_db_count += 1
                    continue # Skip this post

                # Convert cleaned HTML to Markdown (only if cleaning succeeded and body existed)
                # If we reach here, HTML processing was okay.
                try:
                    full_content_markdown = markdownify(cleaned_html, heading_style="ATX", bullets="-")
                    print("  -> Markdown conversion successful.")
                except Exception as e:
                    logging.error(f"Markdownify conversion failed for post ID {post_id} ('{title[:50]}...'). Error: {e}", exc_info=True)
                    print(f"  ERROR: Markdown conversion failed: {e}. Marking for skip and continuing.")
                    record_skip(cur, post_id, norm_title, url)
                    conn.commit() # Commit the skip record
                    already_skipped.add(norm_title)
                    total_skipped_recorded_in_db_count += 1
                    continue # Skip this post

                # --- 6d. Perform Gemini Analyses (only if content available and markdown conversion succeeded) ---
                # If we reach here, full_content_markdown should be populated and valid.
                print("  -> Performing Gemini analyses...")

                # 1. Sentence Summary
                print("    - Generating sentence summary...")
                sentence_summary = summarize_text(full_content_markdown)
                if sentence_summary.startswith("Error:") or sentence_summary.startswith("Analysis skipped") or sentence_summary == "Content was empty.":
                    print(f"    - Sentence summary failed or skipped: {sentence_summary}")
                    logging.warning(f"Sentence summary failed/skipped for post ID {post_id}: {sentence_summary}")
                    sentence_summary = None # Set to None for DB
                    analysis_successful_flag = False # Mark overall analysis as failed for this post
                else:
                    print("    - Sentence summary generated.")

                # 2. Paragraph Summary (Proceed even if previous failed, but flag overall failure)
                print("    - Generating paragraph summary...")
                paragraph_summary = generate_paragraph_summary(full_content_markdown)
                if paragraph_summary.startswith("Error:") or paragraph_summary.startswith("Analysis skipped") or paragraph_summary == "Content was empty.":
                    print(f"    - Paragraph summary failed or skipped: {paragraph_summary}")
                    logging.warning(f"Paragraph summary failed/skipped for post ID {post_id}: {paragraph_summary}")
                    paragraph_summary = None
                    analysis_successful_flag = False
                else:
                    print("    - Paragraph summary generated.")

                # 3. Key Implication (Proceed even if previous failed)
                print("    - Generating key implication...")
                key_implication = generate_key_implication(full_content_markdown)
                if key_implication.startswith("Error:") or key_implication.startswith("Analysis skipped") or key_implication == "Content was empty.":
                    print(f"    - Key implication failed or skipped: {key_implication}")
                    logging.warning(f"Key implication failed/skipped for post ID {post_id}: {key_implication}")
                    key_implication = None
                    analysis_successful_flag = False
                else:
                    print("    - Key implication generated.")

                # 4. Cluster Tag (Proceed even if previous failed)
                print("    - Generating cluster and tags...")
                original_tags = [t.get("name", "N/A") for t in post.get("tags", []) if t]
                cluster_info = generate_cluster_tag(title, original_tags, full_content_markdown)
                if isinstance(cluster_info, dict) and "error" in cluster_info:
                     print(f"    - Cluster/tag generation failed or skipped: {cluster_info['error']}")
                     logging.warning(f"Cluster/tag generation failed/skipped for post ID {post_id}: {cluster_info['error']}")
                     # db_cluster and db_tags remain None
                     analysis_successful_flag = False
                elif isinstance(cluster_info, dict): # Check it's a dict (already validated in function)
                     db_cluster = cluster_info.get("cluster") # Already validated as string
                     db_tags = cluster_info.get("tags")     # Already validated as list of strings
                     # Additional validation for None values (though unlikely)
                     if db_cluster is None or db_tags is None:
                         logging.warning(f"Cluster/tag generation returned None values unexpectedly for post ID {post_id}. Cluster: {db_cluster}, Tags: {db_tags}")
                         analysis_successful_flag = False
                     else:
                         print(f"    - Cluster/tags generated: Cluster='{db_cluster}', Tags={db_tags}")
                else: # Should not happen due to validation in generate_cluster_tag, but handle defensively
                    logging.warning(f"Cluster/tag generation returned unexpected type for post ID {post_id}: {type(cluster_info)}")
                    analysis_successful_flag = False

                # Increment count if any analysis step failed for this post
                if not analysis_successful_flag:
                    failed_analysis_count += 1

                # --- 6f. Extract Other Metadata ---
                print("  -> Extracting remaining metadata...")
                source_type = post.get('source_type', 'Unknown')
                score = post.get('baseScore') # Keep as number (or None)
                comment_count = safe_int_or_zero(post.get('commentCount'))

                # Extract image URL (simple regex for first src or data-src)
                image_url = None
                if cleaned_html: # Use cleaned HTML to avoid script/style interference
                    match = re.search(r'<img[^>]+(?:src|data-src)=["\']([^"\']+)["\']', cleaned_html, re.IGNORECASE)
                    if match:
                        image_url = match.group(1)
                        print(f"    - Found image URL: {image_url[:60]}...")

                # Extract authors (using set for uniqueness)
                authors_set = set()
                if post.get('user') and post['user'].get('displayName'):
                    authors_set.add(post['user']['displayName'])
                if post.get('coauthors'):
                    for author in post['coauthors']:
                        if author and author.get('displayName'):
                            authors_set.add(author['displayName'])
                authors_set.discard(None) # Remove potential None values
                # Convert set to sorted list for consistent DB insertion (ARRAY type)
                authors_list = sorted(list(authors_set)) if authors_set else ['Unknown']
                print(f"    - Authors: {authors_list}")

                # Extract and parse publication date
                published_date = iso_to_dt(post.get('postedAt'))
                print(f"    - Published Date: {published_date}")

                # --- 6e. Generate Embeddings (After Gemini, uses title & results) ---
                print("  -> Generating OpenAI embeddings...")
                # Prepare text inputs for embeddings
                short_text_input = title or "" # Use title for short embedding

                # Combine title and available analysis results for full embedding
                # Use the top-level safe_join helper
                full_text_parts = [
                    title,
                    f"Authors: {', '.join(authors_list)}", # <<< FORMATTED AUTHORS USED HERE
                    sentence_summary,
                    paragraph_summary,  
                    key_implication
                ]
                full_text_input = safe_join("\n\n", full_text_parts) # Join with double newline

                # Call the embedding function (uses global openai_client)
                embedding_short_vector, embedding_full_vector = generate_embeddings(
                    openai_client, short_text_input, full_text_input
                )

                # Check for embedding failure
                if embedding_short_vector is None or embedding_full_vector is None:
                    print(f"    - Embedding generation failed or skipped for post ID {post_id}.")
                    logging.warning(f"Embedding generation failed/skipped for post ID {post_id}")
                    embedding_failures_count += 1
                    # Keep vectors as None, don't mark analysis_successful_flag as False here
                    # as Gemini analysis might have succeeded.
                else:
                    print("    - Embeddings generated successfully.")

                # --- 6g. Prepare Data Tuple for Insertion ---
                # Ensure the order matches DB_COLS exactly!
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
                    html_body,                      # full_content (original HTML, maybe with prepended title)
                    full_content_markdown,          # full_content_markdown (str or None)
                    comment_count,                  # comment_count (int)
                    db_cluster,                     # cluster_tag (AI generated cluster or None)
                    embedding_short_vector,         # embedding_short (list[float] or None)
                    embedding_full_vector           # embedding_full (list[float] or None)
                )

                # <<< ADD THIS DEBUGGING >>>
                # Added flush=True to help ensure output appears before potential crash
                print(f"  DEBUG (Loop Item): Tuple length: {len(data_tuple)}, Expected: {NUM_DB_COLS}", flush=True)
                print(f"  DEBUG (Loop Item): authors_list type: {type(authors_list)}, content: {authors_list}", flush=True)
                print(f"  DEBUG (Loop Item): db_tags type: {type(db_tags)}, content: {db_tags}", flush=True)
                print(f"  DEBUG (Loop Item): embedding_short type: {type(embedding_short_vector)}, len: {len(embedding_short_vector) if embedding_short_vector is not None else 'None'}", flush=True)
                print(f"  DEBUG (Loop Item): embedding_full type: {type(embedding_full_vector)}, len: {len(embedding_full_vector) if embedding_full_vector is not None else 'None'}", flush=True)
                if len(data_tuple) != NUM_DB_COLS:
                    print(f"  ERROR (Loop Item): Tuple length mismatch for post ID {post_id}!", flush=True)
                    print(f"  DEBUG (Loop Item): Tuple content (first 500 chars): {str(data_tuple)[:500]}...", flush=True) # Print partial tuple content
                    # Optionally skip adding the bad tuple:
                    # continue
                # <<< END DEBUGGING >>>

                # --- 6h. Add to Batch ---
                batch_data.append(data_tuple)
                print(f"  -> Added post '{title[:50]}...' to batch (Batch size: {len(batch_data)}).")

                # -------- 7. Insert Batch into Database --------
                if len(batch_data) >= BATCH_SIZE:
                    print(f"\n--- Executing database batch insert ({len(batch_data)} posts) ---")
                    batch_insert_successful = False
                    try:
                        # Use extras.execute_batch for efficient insertion
                        # The INSERT_SQL already handles ON CONFLICT DO NOTHING
                        cur.executemany(INSERT_SQL, batch_data)
                        conn.commit() # Commit the transaction for this batch
                        batch_insert_successful = True
                        print(f"--- Batch insert successful. {len(batch_data)} rows processed (inserted or skipped on conflict). ---")
                        logging.info(f"Successfully executed batch insert for {len(batch_data)} posts.")
                    except psycopg2.DatabaseError as db_err:
                        logging.error(f"Database error during batch execution: {db_err}", exc_info=True)
                        print(f"ERROR: Database error during batch execution: {db_err}. Rolling back batch.")
                        conn.rollback() # Rollback the failed batch transaction
                        total_db_failures += len(batch_data) # Increment failure count
                    except Exception as e:
                        logging.error(f"Unexpected error during batch execution: {e}", exc_info=True)
                        print(f"ERROR: Unexpected error during batch execution: {e}. Rolling back batch.")
                        conn.rollback() # Rollback on unexpected errors
                        total_db_failures += len(batch_data) # Increment failure count
                    finally:
                        # Update total affected rows only if batch succeeded
                        if batch_insert_successful:
                            affected_rows_count += len(batch_data)
                        # Always clear the batch list
                        batch_data = []

            # -------- 8. Insert Final Batch --------
            if batch_data:
                print(f"\n--- Executing final database batch insert ({len(batch_data)} posts) ---", flush=True)
                # <<< ADD DETAILED PRINTING >>>
                print("\\nDEBUG (Final Batch - BEFORE EXECUTION):")
                print(f"  INSERT SQL: {INSERT_SQL}")
                for idx, item_tuple in enumerate(batch_data):
                    print(f"\\n  --- Data Tuple {idx} ---")
                    if len(item_tuple) == NUM_DB_COLS:
                        for col_name, value in zip(DB_COLS, item_tuple):
                            value_str = str(value)
                            print(f"    Column: {col_name}")
                            print(f"      Type: {type(value)}")
                            # Print first 500 chars, add ellipsis if longer
                            print(f"      Value (≤500 chars): {value_str[:500]}{'...' if len(value_str) > 500 else ''}")
                    else:
                         print(f"    ERROR: Tuple length mismatch! Expected {NUM_DB_COLS}, got {len(item_tuple)}")
                         print(f"    Raw Tuple (≤500 chars): {str(item_tuple)[:500]}...")
                print("DEBUG (Final Batch - END PRINTING)\\n")
                # <<< END DETAILED PRINTING >>>
                final_batch_successful = False
                try:
                    cur.executemany(INSERT_SQL, batch_data)
                    conn.commit() # Commit the final batch
                    final_batch_successful = True
                    print(f"--- Final batch insert successful. {len(batch_data)} rows processed. ---")
                    logging.info(f"Successfully executed final batch insert for {len(batch_data)} posts.")
                except psycopg2.DatabaseError as db_err:
                    logging.error(f"Database error during final batch execution: {db_err}", exc_info=True)
                    print(f"ERROR: Database error during final batch execution: {db_err}. Rolling back batch.")
                    conn.rollback()
                    total_db_failures += len(batch_data)
                except Exception as e:
                    logging.error(f"Unexpected error during final batch execution: {e}", exc_info=True)
                    print(f"ERROR: Unexpected error during final batch execution: {e}. Rolling back batch.")
                    conn.rollback()
                    total_db_failures += len(batch_data)
                finally:
                    if final_batch_successful:
                        affected_rows_count += len(batch_data)
                    batch_data = [] # Clear list

    except psycopg2.OperationalError as e:
        # Errors during connection itself
        logging.critical(f"FATAL: Database connection failed: {e}", exc_info=True)
        print(f"FATAL ERROR: Database connection failed: {e}")
        # Cannot proceed without DB connection
    except psycopg2.DatabaseError as e:
        # Other database errors (e.g., during initial title fetch)
        logging.error(f"Database error occurred outside batch processing: {e}", exc_info=True)
        print(f"ERROR: Database error occurred: {e}")
        if conn:
            conn.rollback() # Rollback any potential changes if connection exists
    except Exception as e:
        # Catch-all for unexpected errors in the main processing block
        logging.error(f"An unexpected error occurred in the main processing loop: {e}", exc_info=True)
        print(f"ERROR: An unexpected error occurred: {e}")
        if conn:
            conn.rollback() # Rollback on general errors too
    finally:
        # -------- 9. Close Database Connection --------
        if conn:
            conn.close()
            print("\nDatabase connection closed.")
            logging.info("Database connection closed.")

    # -------- 10. Print Final Summary --------
    end_time = time.time()
    duration = end_time - start_time
    print("\n================================================================")
    print("--- Processing Summary ---")
    print("================================================================")
    print(f"Total posts initially combined from filtered sources: {initial_filtered_count}")
    print(f"Unique posts after deduplication: {total_unique_count}")
    print(f"Posts processed (attempted analysis/DB insert): {processed_count}")
    print(f"Posts with failed/skipped Gemini analysis step(s): {failed_analysis_count}")
    print(f"Posts with failed/skipped OpenAI embedding generation: {embedding_failures_count}")
    print(f"Posts recorded in 'skipped_posts' table this run: {total_skipped_recorded_in_db_count}") # New summary line
    print(f"Total rows processed in successful DB batches: {affected_rows_count}")
    print(f"Estimated rows in failed DB batches: {total_db_failures}")
    print(f"Script finished at {datetime.now(timezone.utc)}")
    print(f"Total execution time: {duration:.2f} seconds")
    print("================================================================")
    logging.info(f"Script finished. Duration: {duration:.2f}s. Processed: {processed_count}. DB Success: {affected_rows_count}. DB Fail: {total_db_failures}. Analysis Fail: {failed_analysis_count}. Embedding Fail: {embedding_failures_count}. Recorded Skips: {total_skipped_recorded_in_db_count}.")


if __name__ == "__main__":
    main()