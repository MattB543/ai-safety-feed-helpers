#!/usr/bin/env python3
"""
Ingest top‑quality AI‑safety content from EA Forum, LessWrong, and Alignment Forum
into the `posts` table of the AI‑Safety‑Feed database.

This script fetches posts via GraphQL, filters them based on date, tags, score,
and comment counts, performs analysis (summarization, implication identification,
clustering/tagging) using the Gemini API, generates embeddings using OpenAI,
calculates additional metadata (word count, reading time, links), and inserts
the processed data into a PostgreSQL database, handling potential duplicates
based on normalized titles.

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
import uuid # For generating UUIDs

import requests
from markdownify import markdownify
from bs4 import BeautifulSoup
import psycopg2
from psycopg2 import extras # Explicit import for batch insertion
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from google import genai as genai
from google.genai import types # Explicit import for types
from openai import OpenAI
from openai import APIError, RateLimitError

# ================================================================
#                      Environment & Setup
# ================================================================
load_dotenv()  # Load .env BEFORE using env vars

# --- Essential Environment Variables ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DATABASE_URL   = os.environ.get("AI_SAFETY_FEED_DB_URL")
OPENAI_API_KEY = os.environ.get("OPEN_AI_FREE_CREDITS_KEY")

# --- Initial Checks ---
if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not set. Cannot perform analysis.")
    sys.exit(1)

if not DATABASE_URL:
    print("CRITICAL ERROR: AI_SAFETY_FEED_DB_URL environment variable not set. Cannot connect to database.")
    sys.exit(1)

if not OPENAI_API_KEY:
    print("CRITICAL ERROR: OPENAI_API_KEY environment variable not set. Cannot generate embeddings.")
    logging.critical("CRITICAL ERROR: OPENAI_API_KEY environment variable not set. Cannot generate embeddings.")
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ================================================================
#                     Forum‑specific constants
# ================================================================
EA_API_URL = "https://forum.effectivealtruism.org/graphql"
EA_AI_SAFETY_TAG_ID = "oNiQsBHA3i837sySD"

LW_API_URL = "https://www.lesswrong.com/graphql"
LW_AI_SAFETY_TAG_ID = "yBXKqk8wEg6eM8w5y"

AF_API_URL = "https://www.alignmentforum.org/graphql"

DEFAULT_LIMIT = 3000
BATCH_SIZE    = 1

CUTOFF_DATE = datetime(2025, 5, 10, tzinfo=timezone.utc) # Example: only posts from May 2025 onwards

# ================================================================
#                 Filtering Thresholds & Tag Sets
# ================================================================
EA_SCORE_THRESHOLD_HIGH         = 85
EA_COMMENT_THRESHOLD_HIGH_SCORE = 0
EA_SCORE_THRESHOLD_MID          = 65
EA_COMMENT_THRESHOLD_MID_SCORE  = 12

LW_SCORE_THRESHOLD_HIGH         = 85
LW_COMMENT_THRESHOLD_HIGH_SCORE = 0
LW_SCORE_THRESHOLD_MID          = 65
LW_COMMENT_THRESHOLD_MID_SCORE  = 20

AF_SCORE_THRESHOLD_HIGH         = 140
AF_COMMENT_THRESHOLD_HIGH_SCORE = -1
AF_SCORE_THRESHOLD_MID          = 100
AF_COMMENT_THRESHOLD_MID_SCORE  = 20

APRIL_FOOLS_TAGS = {"April Fool's", "April Fools' Day"}
AI_TAGS_LW       = {"AI"}

# ================================================================
#                      Database Configuration
# ================================================================
DB_COLS = (
    "uuid", "published_date", "source_updated_at", "title", "title_norm",
    "generated_title", "source_url", "source_type", "authors_display", "authors_ids",
    "content_snippet", "full_content", "short_summary", "long_summary", "key_implication",
    "why_valuable", "image_url", "score", "comment_count", "views",
    "first_comment_at", "last_activity_at", "score_timeseries", "comment_timeseries",
    "source_tag_ids", "source_tag_names", "feed_cluster", "feed_tags",
    "reading_time_minutes", "word_count", "external_links", "novelty_score",
    "novelty_note", "embedding_short", "embedding_full", "analysis_version",
    # "author_credentials", "audio_url", "generated_image", "image_prompt" are excluded as they are not populated by this script
)
NUM_DB_COLS = len(DB_COLS)

INSERT_SQL = f"""
INSERT INTO posts ({', '.join(DB_COLS)})
VALUES ({', '.join(['%s'] * NUM_DB_COLS)})
ON CONFLICT (title_norm) DO NOTHING;
"""

SKIP_INSERT_SQL = """
INSERT INTO skipped_posts (post_id, title_norm, source_url)
VALUES (%s, %s, %s)
ON CONFLICT (title_norm) DO NOTHING;
"""

def record_skip(cur, post_id: str, title_norm: str, source_url: str | None):
    cur.execute(SKIP_INSERT_SQL, (
        str(post_id) if post_id is not None else 'N/A',
        str(title_norm),
        str(source_url) if source_url is not None else None
    ))

# ================================================================
#                          Gemini Helpers
# ================================================================

def call_gemini_api(prompt: str, model_name: str = "gemini-2.5-flash-preview-04-17") -> str: # Updated model
    if not GEMINI_API_KEY:
        logging.warning("call_gemini_api called without GEMINI_API_KEY.")
        return "Analysis skipped (missing API key)."

    logging.debug(f"Calling Gemini API (model: {model_name}) with prompt (first 100 chars): {prompt[:100]}...")
    try:
        # Use the working client-based approach from the old script
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )

        if hasattr(response, 'text'):
            result = response.text.strip()
        elif hasattr(response, 'parts') and response.parts:
            result = "".join(part.text for part in response.parts).strip()
        else:
            logging.warning(f"Gemini API response structure was unexpected: {response}")
            result = "Error: Could not extract text from API response."

        if not result:
             logging.warning(f"Gemini API returned an empty result for prompt: {prompt[:100]}...")
             result = "Error: Analysis returned empty result."
        elif result.startswith("Error:"):
             logging.warning(f"Gemini API returned an error message: '{result}'")

        logging.debug(f"Gemini API call successful. Result (first 100 chars): {result[:100]}...")
        return result

    except types.generation_types.BlockedPromptException as e:
        logging.error(f"Gemini API call failed due to blocked prompt: {e}")
        return "Error: Analysis blocked due to prompt content."
    except types.generation_types.StopCandidateException as e:
        logging.error(f"Gemini API call failed due to stop candidate: {e}")
        return "Error: Analysis stopped unexpectedly by the model."
    except Exception as e:
        logging.error(f"Unexpected error during Gemini API call: {e}", exc_info=True)
        return f"Error during analysis: {e}"

def generate_short_summary(text_to_summarize: str) -> str: # Renamed from summarize_text
    if not text_to_summarize or text_to_summarize.isspace():
        logging.info("Skipping short summary: Input content was empty.")
        return "Content was empty."
    prompt = f"""
Summarize the following AI safety content in 1-2 concise sentences (maximum 50 words).
Focus on the core argument, key insight, or main conclusion.
Use clear, accessible language.
The summary should help readers quickly understand what makes this content valuable.

--- Content to summarize ---
{text_to_summarize}
"""
    return call_gemini_api(prompt)

def generate_long_summary(text_to_summarize: str) -> str: # Renamed from generate_paragraph_summary
    if not text_to_summarize or text_to_summarize.isspace():
        logging.info("Skipping long summary: Input content was empty.")
        return "Content was empty."
    prompt = f"""
Generate a structured summary of the following AI safety content. The summary should consist of:

1.  A brief 1-sentence introduction highlighting the main point.
2.  3-5 bullet points covering key arguments, evidence, or insights. Format EACH bullet point as:
    *   **Key concept**: Explanation.
3.  A brief 1-sentence conclusion with the author's final thoughts.

--- Rules ---
-   Make each bullet point concise.
-   Bold only the key concept at the start of each bullet.
-   Use markdown for bullet points.
-   Include a double line break after the introduction and before the conclusion.
-   Output only the summary.

--- Content to summarize ---
{text_to_summarize}
"""
    return call_gemini_api(prompt)

def generate_key_implication_text(text_to_analyze: str) -> str: # Renamed from generate_key_implication
    if not text_to_analyze or text_to_analyze.isspace():
        logging.info("Skipping key implication: Input content was empty.")
        return "Content was empty."
    prompt = f"""
Based on the AI safety content below, identify the single most important logical consequence or implication in one concise sentence (25-35 words). Focus on:
-   What change in thinking or strategy follows from this content?
-   How might this alter understanding of AI safety or governance?
-   A specific actionable insight.
-   The "so what" for an informed AI safety community member.
The implication should be a direct consequence, not a restatement.

--- Content to analyze ---
{text_to_analyze}
"""
    return call_gemini_api(prompt)

def generate_feed_cluster_and_tags(title: str, tags_list: list[str], content_markdown: str) -> dict: # Renamed
    if not content_markdown or content_markdown.isspace():
        return {"error": "Content was empty."}
    if not title: title = "Untitled"
    if not tags_list: tags_list = ["N/A"]

    prompt = f"""
You are the "AI-Safety-Tagger"—an expert taxonomist for an AI-safety news feed.

---  TASK  ---
Given one blog-style post, do BOTH of the following:

1. **Pick exactly one "Cluster"** that best captures the *main theme*
   (see the list of Clusters below).

2. **Choose 1 to 4 "Canonical Tags"** from the same list that most precisely
   describe the post.
   • Tags *must* come from the taxonomy.
   • Prefer the most specific tags.
   • A tag may be selected even if it appears only in the "Synonyms"
     column—use its Canonical form in your answer.

Return your answer as valid JSON, with this schema:

{{
  "cluster": "<one Cluster name>",
  "tags": ["<Canonical tag 1>", "... up to 4"]
}}

Do not output anything else.

--- INPUT ---
Title: {title}
Original author-supplied tags: {tags_list}
Markdown body:
{content_markdown}

--- TAXONOMY ---
[... Same taxonomy as before ...]
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
- Benchmarks & evals
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

    if raw_response.startswith("Error:") or raw_response.startswith("Analysis skipped"):
        return {"error": raw_response}

    cleaned_response = raw_response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[len("```json"):].strip()
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-len("```")].strip()

    try:
        parsed_json = json.loads(cleaned_response)
        if isinstance(parsed_json, dict) and \
           "cluster" in parsed_json and isinstance(parsed_json["cluster"], str) and \
           "tags" in parsed_json and isinstance(parsed_json["tags"], list) and \
           all(isinstance(tag, str) for tag in parsed_json["tags"]):
            
            cleaned_cluster = remove_parentheses_content(parsed_json["cluster"])
            cleaned_tags = [remove_parentheses_content(tag) for tag in parsed_json["tags"]]
            
            return {"cluster": cleaned_cluster, "tags": cleaned_tags}
        else:
            logging.warning(f"Parsed JSON from cluster/tag API has unexpected structure/types: {parsed_json}")
            return {"error": "Parsed JSON has unexpected structure or types"}
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from cluster/tag API. Error: {e}. Response: '{cleaned_response}'")
        return {"error": f"Failed to parse JSON response: {e}"}
    except Exception as e:
        logging.error(f"Unexpected error processing cluster/tag API response: {e}", exc_info=True)
        return {"error": f"Unexpected error processing cluster/tag response: {e}"}

# ================================================================
#                     OpenAI Embedding Helper
# ================================================================

def generate_embeddings(openai_client, short_text: str, full_text: str, model="text-embedding-3-small") -> tuple[list[float] | None, list[float] | None]:
    if not openai_client:
        logging.warning("OpenAI client not initialized. Skipping embedding generation.")
        return None, None

    short_text = short_text or ""
    full_text = full_text or ""

    if not short_text.strip() and not full_text.strip():
        logging.debug("Skipping embedding generation: Both short and full texts are empty.")
        return None, None
    
    inputs_to_embed = []
    if short_text.strip():
        inputs_to_embed.append(short_text)
    else: # Need to send a placeholder if short_text is empty to maintain response structure
        inputs_to_embed.append("placeholder_for_empty_short_text") 
        
    if full_text.strip():
        inputs_to_embed.append(full_text)
    else: # Placeholder for empty full_text
        inputs_to_embed.append("placeholder_for_empty_full_text")

    try:
        logging.debug(f"  -> Generating OpenAI embeddings for {len(inputs_to_embed)} input(s) using model '{model}'...")
        response = openai_client.embeddings.create(
            model=model,
            input=inputs_to_embed
        )
        
        embedding_short = None
        embedding_full = None

        if len(response.data) == 2:
            if short_text.strip(): # Only assign if original text was not empty
                embedding_short = response.data[0].embedding
            if full_text.strip(): # Only assign if original text was not empty
                embedding_full = response.data[1].embedding
            
            logging.debug(f"  -> OpenAI embeddings generated. Short: {'Yes' if embedding_short else 'No'}, Full: {'Yes' if embedding_full else 'No'}")
            return embedding_short, embedding_full
        else:
            logging.warning(f"Unexpected number of embeddings received: {len(response.data)}")
            return None, None
            
    except (APIError, RateLimitError) as e:
        logging.error(f"OpenAI API error during embedding generation: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error during OpenAI embedding generation: {e}", exc_info=True)
        return None, None

# ================================================================
#                         Utility Helpers
# ================================================================
def generate_uuid_str() -> str:
    """Generates a new UUID and returns it as a string."""
    return str(uuid.uuid4())

def remove_parentheses_content(text: str) -> str:
    if not text: return text
    cleaned = re.sub(r'\([^)]*\)', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def normalise_title(title: str) -> str:
    """
    Normalizes a title: lowercase, removes non-alphanumeric (keeps spaces),
    replaces multiple spaces with one, strips.
    """
    if not title: return ""
    # Convert to lowercase
    title = title.lower()
    # Remove characters that are not letters, numbers, or whitespace
    title = re.sub(r'[^a-z0-9\s]+', '', title)
    # Replace multiple whitespace characters with a single space
    title = re.sub(r"\s+", " ", title).strip()
    return title

def iso_to_dt(iso_string: str | None) -> datetime | None:
    if not iso_string: return None
    try:
        dt_obj = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt_obj.astimezone(timezone.utc) if dt_obj.tzinfo else dt_obj.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse ISO date string: '{iso_string}'. Error: {e}")
        return None

def safe_int_or_none(value: any) -> int | None: # Modified to return None
    if value is None: return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def calculate_word_count_and_reading_time(markdown_text: str | None) -> tuple[int | None, int | None]:
    """Calculates word count and estimated reading time (words/200)."""
    if not markdown_text or markdown_text.isspace():
        return None, None
    
    # Basic word count: split by whitespace
    words = markdown_text.split()
    word_count = len(words)
    
    if word_count == 0:
        return 0, 0
        
    # Average reading speed: 200 words per minute
    reading_time_minutes = round(word_count / 200)
    if reading_time_minutes == 0 and word_count > 0 : # Ensure at least 1 min for very short texts with content
        reading_time_minutes = 1

    return word_count, reading_time_minutes

def extract_external_links(html_content: str | None) -> list[str] | None:
    """Extracts all unique external links (href) from HTML content."""
    if not html_content:
        return None
    
    links = set()
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Basic check for external links (starts with http/https)
            if href and (href.startswith('http://') or href.startswith('https://')):
                links.add(href)
    except Exception as e:
        logging.warning(f"Failed to parse HTML for link extraction: {e}")
        return None # Or empty list: []
        
    return sorted(list(links)) if links else None


# ================================================================
#                      GraphQL Fetching Logic
# ================================================================
POST_FIELDS = """
  _id
  title
  pageUrl
  commentCount
  baseScore
  postedAt
  htmlBody
  createdAt      # For source_updated_at (changed from updatedAt which doesn't exist)
  # excerpt        # Removed - this field doesn't exist on Post type
  # --- Fields below were likely causing the 400 Bad Request or are uncertain ---
  # If these are needed, their exact names on the Post object must be verified
  # via GraphQL schema introspection for each forum.
  # viewCount
  # wordCount
  # firstCommentedAt
  # lastActivityAt
  tags {
    _id          # For source_tag_ids
    name         # For source_tag_names
  }
  user {
    _id          # For authors_ids (Added _id here, usually safe)
    displayName
  }
  coauthors {
    _id          # For authors_ids (Added _id here, usually safe)
    displayName
  }
"""

def get_forum_posts(api_url: str, tag_id: str | None = None, limit: int = DEFAULT_LIMIT) -> list[dict]:
    if tag_id:
        view_clause = f'view: "tagById", tagId: "{tag_id}"'
        source_desc = f"{api_url} (Tag ID: {tag_id})"
    else:
        view_clause = 'view: "top"'
        source_desc = f"{api_url} (View: top)"

    query = f"""
    {{
      posts(
        input: {{
          terms: {{
            {view_clause}
            limit: {limit}
          }}
        }}
      ) {{
        results {{
          {POST_FIELDS}
        }}
      }}
    }}
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AI-Safety-Feed-Ingestion-Script/1.2" # Incremented version
    }

    print(f"Executing GraphQL query for {limit} posts from {source_desc}...")
    logging.info(f"Executing GraphQL query for {limit} posts from {source_desc}")

    try:
        response = requests.post(api_url, json={"query": query}, headers=headers, timeout=90)
        response.raise_for_status() # This is where the HTTPError was raised
        result = response.json()

        if "errors" in result:
            # Log GraphQL-specific errors if the HTTP request was successful but the query failed
            logging.error(f"GraphQL API ({source_desc}) returned errors: {json.dumps(result['errors'], indent=2)}")
            print(f"ERROR: GraphQL API ({source_desc}) returned errors. Check logs. Query: {query}") # Print query for debug
            return []

        if "data" not in result or "posts" not in result["data"] or "results" not in result["data"]["posts"]:
             logging.warning(f"Unexpected response structure from {source_desc}. 'data.posts.results' not found.")
             print(f"WARNING: Unexpected response structure from {source_desc}. Check logs.")
             return []

        posts_data = result["data"]["posts"]["results"]
        print(f"Successfully fetched {len(posts_data)} posts from {source_desc}.")
        logging.info(f"Successfully fetched {len(posts_data)} posts from {source_desc}.")
        return posts_data

    except requests.exceptions.HTTPError as http_err: # Catch HTTPError specifically
        logging.error(f"Query failed for {source_desc} with HTTPError: {http_err}", exc_info=True)
        logging.error(f"Failed GraphQL Query was:\n{query}") # Log the query that failed
        if http_err.response is not None:
            logging.error(f"Response content from server: {http_err.response.text}")
        print(f"ERROR: Query failed for {source_desc}: {http_err}. Check logs for query and response.")
        return []
    except requests.exceptions.Timeout:
        logging.error(f"Query timed out for {source_desc}.")
        print(f"ERROR: Query failed for {source_desc}: Request timed out.")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Query failed for {source_desc}: {e}", exc_info=True)
        print(f"ERROR: Query failed for {source_desc}: {e}")
        return []
    except json.JSONDecodeError as e: # If response is not JSON despite 200 OK (unlikely here after raise_for_status)
        logging.error(f"Failed to decode JSON from {source_desc}: {e}")
        print(f"ERROR: Failed to decode JSON response from {source_desc}. Check logs.")
        return []
    except Exception as e:
        logging.error(f"Unexpected error fetching posts from {source_desc}: {e}", exc_info=True)
        print(f"ERROR: An unexpected error occurred fetching posts from {source_desc}. Check logs.")
        return []

# ================================================================
#                     Source‑Specific Filtering
# ================================================================
# (Filtering logic remains largely the same, as it's pre-DB schema changes)

def filter_ea_posts(posts: list[dict], tag_id: str) -> list[dict]:
    print(f"\n--- Filtering {len(posts)} EA Forum posts ---")
    if not posts: return []
    filtered_posts = []
    for p in posts:
        if not p or not p.get('pageUrl') or not p.get('postedAt'): continue
        posted_at_dt = iso_to_dt(p.get("postedAt"))
        if not posted_at_dt or posted_at_dt < CUTOFF_DATE: continue
        post_tags = p.get("tags", [])
        has_ai_safety_tag = any(t and t.get("_id") == tag_id for t in post_tags)
        has_april_fools_tag = any(t and t.get("name") in APRIL_FOOLS_TAGS for t in post_tags)
        if not has_ai_safety_tag or has_april_fools_tag: continue
        score = safe_int_or_none(p.get("baseScore")) or 0 # Default to 0 if None for comparison
        comments = safe_int_or_none(p.get("commentCount")) or 0
        passes_threshold = (
            (score >= EA_SCORE_THRESHOLD_HIGH and comments >= EA_COMMENT_THRESHOLD_HIGH_SCORE) or
            (score >= EA_SCORE_THRESHOLD_MID and comments >= EA_COMMENT_THRESHOLD_MID_SCORE) # Changed to >=
        )
        if passes_threshold:
            p["source_type"] = "EA Forum"
            filtered_posts.append(p)
    print(f"--- Found {len(filtered_posts)} EA Forum posts meeting criteria ---")
    return filtered_posts

def filter_lw_posts(posts: list[dict]) -> list[dict]:
    print(f"\n--- Filtering {len(posts)} LessWrong posts ---")
    if not posts: return []
    filtered_posts = []
    for p in posts:
        if not p or not p.get('pageUrl') or not p.get('postedAt'): continue
        posted_at_dt = iso_to_dt(p.get("postedAt"))
        if not posted_at_dt or posted_at_dt < CUTOFF_DATE: continue
        tag_names = {t.get("name") for t in p.get("tags", []) if t and t.get("name")}
        has_ai_tag = bool(AI_TAGS_LW & tag_names)
        has_april_fools_tag = bool(APRIL_FOOLS_TAGS & tag_names)
        if not has_ai_tag or has_april_fools_tag: continue
        score = safe_int_or_none(p.get("baseScore")) or 0
        comments = safe_int_or_none(p.get("commentCount")) or 0
        passes_threshold = (
            (score >= LW_SCORE_THRESHOLD_HIGH and comments >= LW_COMMENT_THRESHOLD_HIGH_SCORE) or
            (score >= LW_SCORE_THRESHOLD_MID and comments >= LW_COMMENT_THRESHOLD_MID_SCORE) # Changed to >=
        )
        if passes_threshold:
            p["source_type"] = "Less Wrong"
            filtered_posts.append(p)
    print(f"--- Found {len(filtered_posts)} LessWrong posts meeting criteria ---")
    return filtered_posts

def filter_af_posts(posts: list[dict]) -> list[dict]:
    print(f"\n--- Filtering {len(posts)} Alignment Forum posts ---")
    if not posts: return []
    filtered_posts = []
    for p in posts:
        if not p or not p.get('pageUrl') or not p.get('postedAt'): continue
        posted_at_dt = iso_to_dt(p.get("postedAt"))
        if not posted_at_dt or posted_at_dt < CUTOFF_DATE: continue
        tag_names = {t.get("name") for t in p.get("tags", []) if t and t.get("name")}
        has_april_fools_tag = bool(APRIL_FOOLS_TAGS & tag_names)
        if has_april_fools_tag: continue
        score = safe_int_or_none(p.get("baseScore")) or 0
        comments = safe_int_or_none(p.get("commentCount")) or 0
        passes_threshold = (
            (score >= AF_SCORE_THRESHOLD_HIGH and comments >= AF_COMMENT_THRESHOLD_HIGH_SCORE) or # Changed to >=
            (score >= AF_SCORE_THRESHOLD_MID and comments >= AF_COMMENT_THRESHOLD_MID_SCORE) # Changed to >=
        )
        if passes_threshold:
            p["source_type"] = "Alignment Forum"
            filtered_posts.append(p)
    print(f"--- Found {len(filtered_posts)} Alignment Forum posts meeting criteria ---")
    return filtered_posts

# ================================================================
#                     Deduplication Helper
# ================================================================
def choose_highest_score(posts: list[dict]) -> list[dict]:
    print(f"\n--- Deduplicating {len(posts)} posts by normalized title (keeping highest score) ---")
    posts_by_norm_title: dict[str, dict] = {}
    discarded_no_title = 0
    valid_post_count = 0

    for p in posts:
        title = p.get("title")
        if not title:
            discarded_no_title += 1
            continue
        valid_post_count +=1
        norm_title = normalise_title(title)
        current_score = safe_int_or_none(p.get("baseScore")) or -float('inf') # Handle None for comparison

        existing_post = posts_by_norm_title.get(norm_title)
        if existing_post is None or current_score > (safe_int_or_none(existing_post.get("baseScore")) or -float('inf')):
            posts_by_norm_title[norm_title] = p
    
    unique_posts = list(posts_by_norm_title.values())
    duplicates_removed = valid_post_count - len(unique_posts)
    if discarded_no_title > 0:
        print(f"--- Discarded {discarded_no_title} posts lacking a title ---")
    print(f"--- Kept {len(unique_posts)} unique posts (removed {duplicates_removed} lower-scoring duplicates) ---")
    return unique_posts

# ================================================================
#                     Main Processing Logic
# ================================================================
def main():
    start_time = time.time()
    print(f"Starting AI Safety Feed Ingestion Script at {datetime.now(timezone.utc)}")

    ea_raw = get_forum_posts(EA_API_URL, EA_AI_SAFETY_TAG_ID, limit=DEFAULT_LIMIT)
    lw_raw = get_forum_posts(LW_API_URL, LW_AI_SAFETY_TAG_ID, limit=DEFAULT_LIMIT)
    af_raw = get_forum_posts(AF_API_URL, limit=DEFAULT_LIMIT)

    ea_posts = filter_ea_posts(ea_raw, EA_AI_SAFETY_TAG_ID)
    lw_posts = filter_lw_posts(lw_raw)
    af_posts = filter_af_posts(af_raw)

    combined_filtered_posts = ea_posts + lw_posts + af_posts
    initial_filtered_count = len(combined_filtered_posts)
    print(f"\n--- Total posts after initial filtering: {initial_filtered_count} ---")

    unique_posts = choose_highest_score(combined_filtered_posts)
    total_unique_count = len(unique_posts)

    if not unique_posts:
        print("\nNo unique posts to process. Exiting.")
        return

    conn = None
    processed_count = 0
    affected_rows_count = 0
    failed_analysis_count = 0
    batch_data = []
    total_db_failures = 0
    embedding_failures_count = 0
    total_skipped_recorded_in_db_count = 0

    try:
        print("\n--- Connecting to Database ---")
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        register_vector(conn)
        print("Database connection successful.")

        print("Initializing OpenAI client...")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logging.info("OpenAI client initialized.")

        with conn.cursor() as cur:
            print("Fetching existing normalized titles from 'posts' table...")
            cur.execute("SELECT title_norm FROM posts") # Updated table name
            existing_titles = {row[0] for row in cur.fetchall()}
            print(f"--> Found {len(existing_titles):,} existing titles.")

            print("Fetching already skipped normalized titles...")
            cur.execute("SELECT title_norm FROM skipped_posts")
            already_skipped = {r[0] for r in cur.fetchall()}
            print(f"--> Found {len(already_skipped):,} already skipped titles.")

            print(f"\n--- Starting Processing for {total_unique_count} Unique Posts ---")
            for i, post_data_raw in enumerate(unique_posts):
                processed_count += 1
                post_id_source = post_data_raw.get('_id', 'N/A')
                title = post_data_raw.get('title', 'Untitled')
                source_url = post_data_raw.get('pageUrl', 'N/A')
                print(f"\n[{processed_count}/{total_unique_count}] Processing: '{title[:70]}...' (ID: {post_id_source})")

                norm_title = normalise_title(title)
                if norm_title in already_skipped:
                    print(f"  -> Skipping (already marked as skipped): '{norm_title}'.")
                    continue
                if norm_title in existing_titles:
                    print(f"  -> Skipping (already in 'posts' table): '{norm_title}'.")
                    continue

                # --- Initialize data for the new schema ---
                post_uuid = generate_uuid_str()
                generated_title = title # As per discussion, use original title for now
                
                html_body = post_data_raw.get('htmlBody', '') or ""
                cleaned_html_for_processing = ""
                full_content_md = None # This will be the markdown content for DB

                # --- Clean HTML and Convert to Markdown ---
                if html_body:
                    try:
                        soup = BeautifulSoup(html_body, 'html.parser')
                        for element in soup(["script", "style", "noscript"]): element.decompose()
                        cleaned_html_for_processing = str(soup) # Used for link extraction
                        full_content_md = markdownify(cleaned_html_for_processing, heading_style="ATX", bullets="-")
                        print("  -> HTML cleaned and Markdown conversion successful.")
                    except Exception as e:
                        logging.error(f"HTML/Markdown processing failed for '{title[:50]}...': {e}", exc_info=True)
                        record_skip(cur, post_id_source, norm_title, source_url)
                        conn.commit(); already_skipped.add(norm_title); total_skipped_recorded_in_db_count += 1
                        continue
                else: # No HTML body
                    print("  -> Post has no HTML body. Some features like full_content, word_count will be empty.")
                    # We might still want to process posts with no body if they have titles/tags for other analyses
                    # If no body means skip, then add:
                    # record_skip(cur, post_id_source, norm_title, source_url)
                    # conn.commit(); already_skipped.add(norm_title); total_skipped_recorded_in_db_count += 1
                    # continue
                    full_content_md = "" # Ensure it's an empty string not None if proceeding

                # --- Calculate Word Count & Reading Time ---
                word_count, reading_time_minutes = calculate_word_count_and_reading_time(full_content_md)
                print(f"  -> Word Count: {word_count}, Reading Time: {reading_time_minutes} min")

                # --- Extract External Links ---
                external_links = extract_external_links(cleaned_html_for_processing)
                print(f"  -> External Links: {len(external_links) if external_links else 0} found")
                
                # --- Perform Gemini Analyses ---
                analysis_this_post_failed = False
                print("  -> Performing Gemini analyses...")
                short_summary_text = generate_short_summary(full_content_md)
                if short_summary_text.startswith("Error:") or short_summary_text == "Content was empty.":
                    analysis_this_post_failed = True; short_summary_text = None
                
                long_summary_text = generate_long_summary(full_content_md)
                if long_summary_text.startswith("Error:") or long_summary_text == "Content was empty.":
                    analysis_this_post_failed = True; long_summary_text = None

                key_implication_text = generate_key_implication_text(full_content_md)
                if key_implication_text.startswith("Error:") or key_implication_text == "Content was empty.":
                    analysis_this_post_failed = True; key_implication_text = None

                source_tags_for_gemini = [t.get("name", "N/A") for t in post_data_raw.get("tags", []) if t]
                cluster_info = generate_feed_cluster_and_tags(title, source_tags_for_gemini, full_content_md)
                feed_cluster_text, feed_tags_list = None, None
                if isinstance(cluster_info, dict) and "error" not in cluster_info:
                    feed_cluster_text = cluster_info.get("cluster")
                    feed_tags_list = cluster_info.get("tags")
                else:
                    analysis_this_post_failed = True
                
                if analysis_this_post_failed:
                    failed_analysis_count += 1
                    print("  -> One or more Gemini analyses failed for this post.")
                else:
                    print("  -> Gemini analyses completed.")

                # --- Extract Other Metadata from Raw Post ---
                source_type = post_data_raw.get('source_type', 'Unknown')
                score_val = safe_int_or_none(post_data_raw.get('baseScore'))
                comment_count_val = safe_int_or_none(post_data_raw.get('commentCount'))
                views_val = safe_int_or_none(post_data_raw.get('viewCount')) # New
                
                published_date_dt = iso_to_dt(post_data_raw.get('postedAt'))
                source_updated_at_dt = iso_to_dt(post_data_raw.get('createdAt')) # New
                first_comment_at_dt = iso_to_dt(post_data_raw.get('firstCommentedAt')) # New
                last_activity_at_dt = iso_to_dt(post_data_raw.get('lastActivityAt')) # New

                content_snippet_text = None  # excerpt field doesn't exist, so set to None

                # Authors
                authors_display_list = []
                authors_ids_list = []
                if post_data_raw.get('user'):
                    if post_data_raw['user'].get('displayName'): authors_display_list.append(post_data_raw['user']['displayName'])
                    if post_data_raw['user'].get('_id'): authors_ids_list.append(post_data_raw['user']['_id'])
                if post_data_raw.get('coauthors'):
                    for coauthor in post_data_raw['coauthors']:
                        if coauthor.get('displayName'): authors_display_list.append(coauthor['displayName'])
                        if coauthor.get('_id'): authors_ids_list.append(coauthor['_id'])
                authors_display_list = sorted(list(set(authors_display_list))) or ['Unknown']
                authors_ids_list = sorted(list(set(authors_ids_list))) or ['unknown_id']


                # Source Tags
                source_tag_ids_list = None
                source_tag_names_list = None
                if post_data_raw.get("tags"):
                    source_tag_ids_list = sorted(list(set(t['_id'] for t in post_data_raw["tags"] if t and t.get('_id'))))
                    source_tag_names_list = sorted(list(set(t['name'] for t in post_data_raw["tags"] if t and t.get('name'))))

                image_url_val = None # Logic for image_url can be re-added if needed from original script
                if cleaned_html_for_processing:
                    match = re.search(r'<img[^>]+(?:src|data-src)=["\']([^"\']+)["\']', cleaned_html_for_processing, re.IGNORECASE)
                    if match: image_url_val = match.group(1)

                # --- Generate Embeddings ---
                print("  -> Generating OpenAI embeddings...")
                # Input for short embedding: generated_title
                # Input for full embedding: short_summary, long_summary, key_implication, feed_tags
                full_text_for_embedding_parts = [
                    short_summary_text or "",
                    long_summary_text or "",
                    key_implication_text or "",
                    ", ".join(feed_tags_list) if feed_tags_list else ""
                ]
                full_text_for_embedding = "\n".join(filter(None, full_text_for_embedding_parts)).strip()

                embedding_short_vec, embedding_full_vec = generate_embeddings(
                    openai_client, generated_title, full_text_for_embedding
                )
                if embedding_short_vec is None and embedding_full_vec is None and (generated_title or full_text_for_embedding): # only count as failure if there was text to embed
                    embedding_failures_count += 1
                    print("    - Embedding generation failed or skipped.")
                else:
                    print("    - Embeddings generated (or skipped if no text).")
                
                # --- Prepare Data Tuple for Insertion (Matches DB_COLS) ---
                data_tuple = (
                    post_uuid, published_date_dt, source_updated_at_dt, title, norm_title,
                    generated_title, source_url, source_type, authors_display_list, authors_ids_list,
                    content_snippet_text, full_content_md, short_summary_text, long_summary_text, key_implication_text,
                    None, # why_valuable (NULL for now)
                    image_url_val, score_val, comment_count_val, views_val,
                    first_comment_at_dt, last_activity_at_dt, 
                    None, None, # score_timeseries, comment_timeseries (JSONB, NULL for now)
                    source_tag_ids_list, source_tag_names_list, feed_cluster_text, feed_tags_list,
                    reading_time_minutes, word_count, external_links,
                    None, None, # novelty_score, novelty_note (NUMERIC, TEXT, NULL for now)
                    embedding_short_vec, embedding_full_vec,
                    "1.1" # analysis_version (updated from 1.0)
                )
                
                if len(data_tuple) != NUM_DB_COLS:
                    logging.error(f"Tuple length mismatch for post ID {post_id_source}! Expected {NUM_DB_COLS}, got {len(data_tuple)}. Skipping this post.")
                    print(f"  ERROR: Tuple length mismatch for post ID {post_id_source}! Check logs. Skipping.")
                    # This is a critical error, should not happen if DB_COLS and tuple are aligned.
                    # Consider adding it to skipped_posts or raising an error.
                    record_skip(cur, post_id_source, norm_title, source_url) # Record as skip
                    conn.commit()
                    already_skipped.add(norm_title)
                    total_skipped_recorded_in_db_count += 1
                    continue


                batch_data.append(data_tuple)
                print(f"  -> Added post '{title[:50]}...' to batch (Size: {len(batch_data)}).")

                if len(batch_data) >= BATCH_SIZE:
                    print(f"\n--- Executing DB batch insert ({len(batch_data)} posts) ---")
                    try:
                        extras.execute_batch(cur, INSERT_SQL, batch_data) # Use execute_batch
                        conn.commit()
                        affected_rows_count += len(batch_data) # Assume all were processed (inserted or skipped by ON CONFLICT)
                        print(f"--- Batch insert successful. ---")
                    except psycopg2.Error as db_err: # Catch any psycopg2 error
                        logging.error(f"DB error during batch: {db_err}", exc_info=True)
                        print(f"ERROR: DB error during batch: {db_err}. Rolling back.")
                        conn.rollback()
                        total_db_failures += len(batch_data)
                    finally:
                        batch_data = []

            if batch_data: # Final batch
                print(f"\n--- Executing final DB batch insert ({len(batch_data)} posts) ---")
                try:
                    extras.execute_batch(cur, INSERT_SQL, batch_data)
                    conn.commit()
                    affected_rows_count += len(batch_data)
                    print(f"--- Final batch insert successful. ---")
                except psycopg2.Error as db_err:
                    logging.error(f"DB error during final batch: {db_err}", exc_info=True)
                    print(f"ERROR: DB error during final batch: {db_err}. Rolling back.")
                    conn.rollback()
                    total_db_failures += len(batch_data)
                finally:
                    batch_data = []

    except psycopg2.OperationalError as e:
        logging.critical(f"FATAL: Database connection failed: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Unexpected error in main processing: {e}", exc_info=True)
        if conn: conn.rollback()
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

    duration = time.time() - start_time
    print("\n================================================================")
    print("--- Processing Summary ---")
    print(f"Total posts initially filtered: {initial_filtered_count}")
    print(f"Unique posts after deduplication: {total_unique_count}")
    print(f"Posts processed (attempted): {processed_count}")
    print(f"Posts with failed Gemini analysis: {failed_analysis_count}")
    print(f"Posts with failed OpenAI embedding: {embedding_failures_count}")
    print(f"Posts recorded in 'skipped_posts': {total_skipped_recorded_in_db_count}")
    print(f"Total rows processed in DB batches (estimate): {affected_rows_count}")
    print(f"Estimated rows in failed DB batches: {total_db_failures}")
    print(f"Total execution time: {duration:.2f} seconds")
    print("================================================================")
    logging.info(f"Script finished. Duration: {duration:.2f}s. Processed: {processed_count}. DB Success: {affected_rows_count}. DB Fail: {total_db_failures}.")

if __name__ == "__main__":
    main()