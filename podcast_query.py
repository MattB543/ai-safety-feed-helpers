"""
Ingest podcast episodes from various RSS feeds relevant to AI Safety
into the `content` table of the AI-Safety-Feed database.

This script fetches episodes via RSS, attempts to transcribe audio using
AssemblyAI if an audio URL is present, and falls back to HTML show notes
if transcription fails or is unavailable. It filters episodes based on date
and an AI safety guardrail, performs analysis (summarization, implication
identification, clustering/tagging using OpenAI GPT-4o; embeddings using OpenAI)
on the available content (transcript preferred), and inserts the processed
data into a PostgreSQL database, handling duplicates based on normalized titles.

*** MODIFIED: Google Cloud Storage and Speech-to-Text transcription code REMOVED. ***
***           AssemblyAI transcription ADDED. HTML show notes used as fallback.   ***
"""

# ================================================================
#                            Imports
# ================================================================
import os
import re
import json
import time
import logging
from datetime import datetime, timezone, timedelta
import sys
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any, Tuple, Set

import feedparser
from bs4 import BeautifulSoup
from markdownify import markdownify
import psycopg2
from psycopg2 import extras # For batch insertion
from psycopg2 import OperationalError as Psycopg2OpError # Alias for clarity
from pgvector.psycopg2 import register_vector # <-- ADDED for pgvector
from dotenv import load_dotenv

# --- AI/ML Libs ---
from openai import OpenAI, APIError as OpenAI_APIError, RateLimitError as OpenAI_RateLimitError
import assemblyai as aai # <-- ADDED
from assemblyai import TranscriptStatus # <-- ADDED

# ================================================================
#                      Environment & Setup
# ================================================================
load_dotenv()  # Load .env BEFORE using env vars

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, # Changed default to INFO
                    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')

# Suppress overly verbose logs from underlying libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("feedparser").setLevel(logging.INFO) # Allow feedparser info logs


# --- Essential Environment Variables ---
DATABASE_URL   = os.environ.get("AI_SAFETY_FEED_DB_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY") # <-- ADDED

# --- Initial Checks ---
if not DATABASE_URL:
    logging.critical("CRITICAL ERROR: AI_SAFETY_FEED_DB_URL environment variable not set. Cannot connect to database.")
    sys.exit(1)
if not OPENAI_API_KEY:
    logging.critical("CRITICAL ERROR: OPENAI_API_KEY environment variable not set. Cannot generate embeddings.")
    sys.exit(1)
if not ASSEMBLYAI_API_KEY: # <-- ADDED Check
    # Make this a warning, not critical, as HTML fallback exists
    logging.warning("WARNING: ASSEMBLYAI_API_KEY environment variable not set. Audio transcription will be skipped.")


# --- Initialize API Clients ---
logging.info("Initializing API clients...")
openai_client = None
# AssemblyAI configuration (done globally)
assemblyai_configured = False #
if ASSEMBLYAI_API_KEY:
    try:
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        assemblyai_configured = True
        logging.info("AssemblyAI client configured.")
    except Exception as e:
        logging.error(f"Error configuring AssemblyAI: {e}. Transcription will be disabled.")
        ASSEMBLYAI_API_KEY = None # Ensure it's None if config fails
else:
    logging.info("AssemblyAI transcription skipped (no API key).")

try:
    # Initialize OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("OpenAI client initialized.")


except Exception as e:
    logging.critical(f"CRITICAL ERROR initializing API clients (OpenAI): {e}", exc_info=True)
    # Decide if script should exit if *any* client fails
    sys.exit(1) # Exit if essential clients fail

logging.info("API client initialization complete.")

# ================================================================
#    OpenAI Chat Helper  (replaces call_gemini_api)
# ================================================================

def call_gpt_api(prompt: str,
                 model: str = "gpt-4.1",      # GPT-4.1 public name
                 temperature: float = 0.2) -> str:
    """
    Sends a single-prompt exchange to OpenAI ChatCompletions and
    returns the content string. Raises exceptions on failure.
    
    Raises:
        ValueError: If OpenAI client not initialized or API returns no content
        OpenAI_APIError: For OpenAI API errors
        OpenAI_RateLimitError: For OpenAI rate limit errors
        Exception: For other unexpected errors
    """
    if not openai_client:
        raise ValueError("OpenAI client not initialized")

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant specialised in AI-safety content analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature)
        
        # Check if choices is not empty and message content exists
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            raise ValueError("OpenAI API returned no content or unexpected response structure")
            
    except (OpenAI_APIError, OpenAI_RateLimitError) as e:
        logging.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during OpenAI API call: {e}", exc_info=True)
        raise

# ---------- Specific Analysis Functions ----------

def summarize_text(text_to_summarize: str) -> str:
    """Generates a concise 1-2 sentence summary using the OpenAI GPT model."""
    if not text_to_summarize or text_to_summarize.isspace():
        logging.info("Skipping sentence summary: Input content was empty.")
        return "Content was empty."
    
    prompt = f"Summarize the following AI safety content in 2 concise sentences (maximum 50 words). Focus on the core argument, key insight, or main conclusion rather than methodology. Use clear, accessible language while preserving technical accuracy. The summary should be very readable and should help readers quickly understand what makes this content valuable or interesting and decide if they want to read more.\n\nContent to summarize:\n{text_to_summarize}"
    
    try:
        return call_gpt_api(prompt)
    except Exception as e:
        logging.warning(f"Sentence summary failed: {type(e).__name__}: {e}")
        return None

def generate_paragraph_summary(text_to_summarize: str) -> str:
    """Generates a structured paragraph summary using the OpenAI GPT model."""
    if not text_to_summarize or text_to_summarize.isspace():
        logging.info("Skipping paragraph summary: Input content was empty.")
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
    
    try:
        return call_gpt_api(prompt)
    except Exception as e:
        logging.warning(f"Paragraph summary failed: {type(e).__name__}: {e}")
        return None

def generate_key_implication(text_to_analyze: str) -> str:
    """Identifies the single most important logical consequence using the OpenAI GPT model."""
    if not text_to_analyze or text_to_analyze.isspace():
        logging.info("Skipping key implication: Input content was empty.")
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
    
    try:
        return call_gpt_api(prompt)
    except Exception as e:
        logging.warning(f"Key implication generation failed: {type(e).__name__}: {e}")
        return None

def generate_cluster_tag(title: str, tags_list: List[str], content_markdown: str) -> Dict[str, Any]:
    """
    Generates a cluster and canonical tags using the OpenAI GPT model.
    Returns: dict: Parsed JSON or {"error": "Reason string"}.
    """
    if not content_markdown or content_markdown.isspace():
        logging.info("Skipping cluster tag generation: Input content was empty.")
        return {"error": "Input content was empty."}

    def remove_parentheses(text: str) -> str:
        """Remove everything in parentheses and the parentheses themselves."""
        if not text:
            return text
        # Use regex to remove parentheses and their contents
        cleaned = re.sub(r'\s*\([^)]*\)\s*', '', text)
        # Clean up any extra whitespace
        return cleaned.strip()

    prompt = f"""
You are the "AI-Safety-Tagger"—an expert taxonomist for an AI-safety news feed.

---  TASK  ---
Given one blog-style post, do BOTH of the following:

1. **Pick exactly one "Cluster"** that best captures the *main theme*
   (see the list of Clusters below).

2. **Choose 1 to 4 "Canonical Tags"** from the same list that most precisely
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
    
    try:
        raw_response = call_gpt_api(prompt)
        # Try to parse the response as JSON
        json_response = json.loads(raw_response)
        # Basic validation
        if not isinstance(json_response, dict):
            return {"error": "Response was not a JSON object"}
        if "cluster" not in json_response or "tags" not in json_response:
            return {"error": "Response missing required fields"}
        if not isinstance(json_response["tags"], list):
            return {"error": "Tags field was not a list"}
        
        # Clean parentheses from cluster and tags
        if json_response.get("cluster"):
            json_response["cluster"] = remove_parentheses(json_response["cluster"])
        
        if json_response.get("tags") and isinstance(json_response["tags"], list):
            json_response["tags"] = [remove_parentheses(tag) for tag in json_response["tags"] if tag]
        
        # Return the cleaned response
        return json_response
    except (ValueError, OpenAI_APIError, OpenAI_RateLimitError) as e:
        logging.warning(f"Cluster/tag generation failed: {type(e).__name__}: {e}")
        return {"error": f"API error: {type(e).__name__}"}
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse cluster/tag JSON response: {e}")
        return {"error": f"JSON parse error: {str(e)}"}
    except Exception as e:
        logging.error(f"Unexpected error processing cluster/tag response: {e}")
        return {"error": f"Processing error: {type(e).__name__}"}

def is_ai_safety_post(title: str, html_body: str) -> bool:
    """
    Fast yes/no guard-rail: returns True iff the post's title and content
    are sufficiently focused on AI safety topics to be included in the feed.
    Uses OpenAI's GPT-4o-mini model for fast classification.
    """
    if not openai_client:
        logging.warning("AI safety guard-rail check skipped: OpenAI client not initialized. Defaulting to True (fail-open).")
        return True

    # Clean and extract text from HTML if present
    text_content = ""
    if html_body:
        try:
            soup = BeautifulSoup(html_body, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)
        except Exception as e:
            logging.warning(f"Failed to extract text from HTML: {e}")
            text_content = html_body  # Fall back to raw HTML

    # Prepare content for analysis (first ~500 chars of text + title)
    analysis_text = f"Title: {title}\n\nContent excerpt: {text_content[:500]}..."

    prompt = f"""
Analyze this podcast episode's title and content excerpt to determine if it is sufficiently focused on AI safety topics to be included in an AI safety content feed.

Content to analyze:
{analysis_text}

Rules for inclusion:
1. The content must substantially discuss AI safety, AI alignment, AI governance, AI policy, etc.
2. Very general AI/ML technical content is NOT sufficient - there must be a clear safety/ethics/governance/etc. angle.
3. Brief mentions of AI safety in otherwise unrelated content is NOT sufficient.

Respond with ONLY "yes" or "no". Use "yes" if you are reasonably confident the content meets the criteria, otherwise use "no".
"""
    
    try:
        response = call_gpt_api(
            prompt=prompt,
            model="gpt-4.1",  # Use the fast model for this guardrail
            temperature=0.1   # Low temperature for more consistent yes/no
        )
        
        # Clean and validate response
        response = response.strip().lower()
        is_relevant = response == "yes"

        # Log the decision
        if is_relevant:
            logging.debug(f"Guard-rail: Content accepted as AI safety relevant.")
        else:
            logging.info(f"Guard-rail: Content rejected as not sufficiently AI safety focused.")

        return is_relevant
        
    except Exception as e:
        logging.warning(f"AI safety guard-rail check failed: {type(e).__name__}: {e}. Defaulting to True (fail-open).")
        return True

# ================================================================
#                         Constants
# ================================================================


# Ingest only posts published on/after this date (UTC, inclusive)
CUTOFF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
# MODIFIED: Target 2 successful insertions per feed
TARGET_INSERTIONS_PER_FEED = 2
# Safety limit: Max posts to *check* per feed if target not reached
MAX_POSTS_PER_FEED = 100

# --- Source Feeds ---
SOURCE_FEEDS = {
    "https://feeds.fame.so/ai-government-and-the-future": "AI, Government, and the Future",
    "https://feeds.feedburner.com/80000HoursPodcast": "80,000 Hours Podcast",
    "https://api.substack.com/feed/podcast/69345.rss": "Dwarkesh Podcast",
    "https://axrp.net/feed.xml": "AXRP",
    "https://anchor.fm/s/1e4a0eac/podcast/rss": "Machine Learning Street Talk",
    "https://futureoflife.org/podcast/feed/": "FLI Podcast",
    "https://samharris.org/subscriber-rss/?uid=PNIypzHCbq78upz": "Making Sense with Sam Harris",
    "https://lexfridman.com/feed/podcast/": "Lex Fridman Podcast",
    "https://anchor.fm/s/ebbe3a98/podcast/rss": "For Humanity: An AI Safety Podcast",
    "https://podcast.clearerthinking.org/rss.xml": "Clearer Thinking",
    "http://feeds.libsyn.com/182816/rss": "Alignment Newsletter Podcast",
    "https://www.machine-ethics.net/itunes-rss-feed/": "Machine Ethics Podcast",
    "https://feeds.transistor.fm/intoaisafety": "Into AI Safety",
    "https://feeds.buzzsprout.com/2210416.rss": "Center for AI Policy Podcast",
    "https://pinecast.com/feed/hear-this-idea": "Hear This Idea",
    "https://feeds.libsyn.com/539322/rss": "AI Governance Podcast",
    "https://pod.link/1526725061.rss": "TechTank Podcast",
    "https://feeds.simplecast.com/9YNI3WaL": "Your Undivided Attention",
    "https://feeds.buzzsprout.com/2172898.rss": "Cognitive Revolution",
}

# ================================================================
#                      Database Configuration
# ================================================================
# Define the columns in the 'content' table that we will insert into
# IMPORTANT: Ensure this order matches the data tuple created later.
# 'title_norm' IS included here, assuming it's a regular column to be inserted.
# If it's a DB-generated column, remove it from DB_COLS and the INSERT statement.
DB_COLS = (
    "source_url", "title", "source_type", "authors", "published_date",
    "topics", "score", "image_url", "sentence_summary", "paragraph_summary",
    "key_implication", "full_content", "full_content_markdown",
    "comment_count", "cluster_tag",
    "embedding_short", "embedding_full", 
    "audio_url",
)
NUM_DB_COLS = len(DB_COLS)

# Pre-compute the INSERT SQL statement for efficiency
# Uses ON CONFLICT with the 'title_norm' column to prevent duplicates
# NOTE: 'title_norm' is still the target for ON CONFLICT, even though it's not in the INSERT list.
INSERT_SQL = f"""
INSERT INTO content ({', '.join(DB_COLS)})
VALUES ({', '.join(['%s'] * NUM_DB_COLS)})
ON CONFLICT (title_norm) DO NOTHING;
"""

SKIP_INSERT_SQL = """
INSERT INTO skipped_posts (post_id, title_norm, source_url)
VALUES (%s, %s, %s)
ON CONFLICT (title_norm) DO NOTHING;
"""

def record_skip(cur, post_id: str, title_norm: str, source_url: Optional[str]):
    """Insert one row into skipped_posts (no commit)."""
    cur.execute(SKIP_INSERT_SQL, (post_id, title_norm, source_url))

# ================================================================
#                          Utility Helpers
# ================================================================

def normalise_title(title: str) -> str:
    """Normalizes a title string to match the database `title_norm` generation.

    This involves:
    1. Lowercasing the title.
    2. Replacing one or more whitespace characters with a single space.
    3. Stripping leading/trailing whitespace.
    """
    if not title: return ""
    # 1. Lowercase
    normalized = title.lower()
    # 2. Replace multiple whitespace chars with a single space
    normalized = re.sub(r'\s+', ' ', normalized)
    # 3. Strip leading/trailing whitespace
    normalized = normalized.strip()
    return normalized

def iso_to_dt(iso_string: Optional[str]) -> Optional[datetime]:
    """Converts ISO 8601 string to timezone-aware datetime object (UTC)."""
    if not iso_string: return None
    try:
        dt_obj = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        if dt_obj.tzinfo is None:
            return dt_obj.replace(tzinfo=timezone.utc)
        return dt_obj.astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse ISO date string: '{iso_string}'. Error: {e}")
        return None

def struct_time_to_dt(st: Optional[time.struct_time]) -> Optional[datetime]:
    """Converts feedparser's time.struct_time to timezone-aware datetime (UTC)."""
    if not st: return None
    try:
        # Assume UTC if feedparser doesn't provide timezone info (common)
        dt_naive = datetime(*st[:6])
        return dt_naive.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, IndexError) as e:
        logging.warning(f"Could not convert time.struct_time {st} to datetime: {e}")
        return None

def safe_int_or_none(value: Any) -> Optional[int]:
    """Safely converts value to int, returning None on failure."""
    if value is None: return None
    try: return int(value)
    except (ValueError, TypeError): return None

def get_hostname_from_url(url: str) -> Optional[str]:
    """Extracts the hostname from a URL."""
    if not url: return None
    try: return urlparse(url).netloc.lower()
    except Exception as e:
        logging.warning(f"Could not parse hostname from URL '{url}': {e}")
        return None

# ================================================================
#                  AssemblyAI Transcription Helper
# ================================================================

def transcribe_audio_assemblyai(audio_url: str, entry_title: str) -> Optional[str]:
    """
    Transcribes audio from a URL using AssemblyAI.

    Args:
        audio_url: The public URL of the audio file.
        entry_title: The title of the entry (for logging).

    Returns:
        The transcribed text as a string, or None if transcription fails,
        is skipped, or the URL is invalid.
    """
    # Check if AssemblyAI was configured successfully during startup
    if not assemblyai_configured:
        logging.info(f"Skipping AssemblyAI transcription for '{entry_title[:50]}...': Client not configured.")
        return None
    if not audio_url:
        logging.debug(f"Skipping AssemblyAI transcription for '{entry_title[:50]}...': No audio URL provided.")
        return None

    # Basic URL validation
    if not urlparse(audio_url).scheme in ['http', 'https']:
         logging.warning(f"Skipping AssemblyAI transcription for '{entry_title[:50]}...': Invalid audio URL scheme ({audio_url[:60]}...).")
         return None

    logging.info(f"  Attempting AssemblyAI transcription for '{entry_title[:50]}...' (URL: {audio_url[:60]}...)")
    start_transcribe_time = time.time()

    try:
        # Configure transcription options
        # You can add more options here like speaker_labels=True, auto_highlights=True etc.
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best # Use 'best' for higher accuracy, 'nano' for speed/cost
            # speaker_labels=True # Uncomment if you want speaker diarization
        )
        transcriber = aai.Transcriber(config=config)

        # The transcribe method with a URL handles polling for completion
        transcript = transcriber.transcribe(audio_url)

        duration = time.time() - start_transcribe_time

        # --- Check Transcript Status ---
        if transcript.status == TranscriptStatus.error.value:
            logging.error(f"  AssemblyAI transcription FAILED for '{entry_title[:50]}...' after {duration:.2f}s. Error: {transcript.error}")
            return None
        elif transcript.status == TranscriptStatus.completed.value:
            logging.info(f"  AssemblyAI transcription SUCCEEDED for '{entry_title[:50]}...' in {duration:.2f}s. Transcript length: {len(transcript.text or '')} chars.")

            # --- Optional: Format with Speaker Labels ---
            # if config.speaker_labels and transcript.utterances:
            #     formatted_text = "\\n".join(
            #         f"Speaker {u.speaker}: {u.text}" for u in transcript.utterances
            #     )
            #     return formatted_text
            # else:
            #     return transcript.text # Return plain text if no speaker labels or utterances

            return transcript.text # Return the plain text transcript

        else:
            # Handle unexpected statuses (queued, processing - though SDK should wait)
            logging.warning(f"  AssemblyAI transcription for '{entry_title[:50]}...' finished with unexpected status: {transcript.status} after {duration:.2f}s.")
            return None

    except aai.ApiError as e:
        # Handle AssemblyAI specific API errors (auth, rate limits, etc.)
        logging.error(f"  AssemblyAI API error during transcription for '{entry_title[:50]}...': {e}", exc_info=False) # Keep log cleaner
        logging.debug(f"AssemblyAI API error details", exc_info=True) # Full trace in debug
        return None
    except Exception as e:
        # Catch potential network errors, timeouts, SDK issues etc.
        logging.error(f"  Unexpected error during AssemblyAI transcription for '{entry_title[:50]}...': {type(e).__name__} - {e}", exc_info=True)
        return None

# ================================================================
#                     OpenAI Embedding Helper
# ================================================================

def generate_embeddings(short_text: str, full_text: str, model="text-embedding-3-small") -> tuple[list[float] | None, list[float] | None]:
    """
    Generates short and full embeddings for the given texts using OpenAI.

    Args:
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
    except (OpenAI_APIError, OpenAI_RateLimitError) as e:
        logging.error(f"OpenAI API error during embedding generation: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error during OpenAI embedding generation: {e}", exc_info=True)
        return None, None

# ================================================================
#                      Feed Fetching Iterator
# ================================================================

def extract_audio_url(entry):
    """Extract audio URL from RSS entry."""
    # Check enclosures
    for enc in entry.get("enclosures", []):
        if isinstance(enc, dict) and enc.get("type", "").startswith("audio/") and enc.get("href"):
            return enc["href"]
    
    # Check links
    for link in entry.get("links", []):
        if isinstance(link, dict) and link.get('rel') == 'enclosure' and link.get('type', '').startswith('audio/') and link.get('href'):
            return link["href"]
    
    # Check media_content
    for media in entry.get("media_content", []):
        if isinstance(media, dict) and media.get("medium") == "audio" and media.get("url"):
            return media["url"]
    
    # Check if main link is audio file
    link_url = entry.get("link", "")
    if link_url and link_url.endswith((".mp3", ".m4a", ".wav")):
        return link_url
    
    return None

def iter_rss_feed(feed_url: str, source_name: str) -> Dict[str, Any]:
    """
    Simplified iterator: Fetches and yields basic info for each entry in an RSS feed.
    Handles feed parsing errors gracefully.
    """
    logging.info(f"Fetching RSS feed: {source_name} ({feed_url})")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }
    feed_data = None
    try:
        # Consider adding timeout via requests if feedparser hangs
        feed_data = feedparser.parse(feed_url, request_headers=headers, agent=headers["User-Agent"])
        if feed_data.bozo:
             logging.warning(f"Feed '{source_name}' is not well-formed: {feed_data.bozo_exception}")
             # Continue processing if entries exist despite bozo flag

    except Exception as e:
         logging.error(f"Fatal error fetching or parsing feed '{source_name}' ({feed_url}): {e}", exc_info=True)
         return # Stop iteration for this feed on fatal error

    if not feed_data or not feed_data.entries:
        status_code = feed_data.get('status', 'N/A') if feed_data else 'N/A'
        logging.warning(f"RSS feed for '{source_name}' is empty or could not be parsed fully. Status: {status_code}. Contains {len(feed_data.get('entries', []))} entries.")
        return # Stop iteration if no entries

    logging.info(f"Found {len(feed_data.entries)} entries in feed '{source_name}'.")

    for i, entry in enumerate(feed_data.entries):
        # --- Basic Data Extraction ---
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        # Use link as fallback ID, ensure it's not empty
        post_id = entry.get("id", link) or f"{link}-{title}" # Create a more robust fallback ID

        if not title or not link or not post_id:
            logging.debug(f"Skipping entry {i+1} from '{source_name}': Missing essential field (title, link, or ID). Title='{title[:50]}...', Link='{link}', ID='{post_id}'")
            continue

        # --- Extract Content (HTML) ---
        html_body = "" 

        # Attempt 1: From entry.content
        content_items = entry.get("content")
        parsed_content_from_list = None # Stores result from this block
        if content_items and isinstance(content_items, list) and len(content_items) > 0:
            # Attempt 1a: specific 'text/html' type
            # Ensure 'item' is a dict before calling .get()
            val_html_type = next(
                (item.get('value', '') for item in content_items if isinstance(item, dict) and item.get('type') == 'text/html'),
                None 
            )
            
            # Attempt 1b: first item in list (this is the fallback value)
            val_first_item = "" # Default if first item is not a dict or has no value
            first_item_obj = content_items[0] # Known to exist due to len check
            if isinstance(first_item_obj, dict):
                val_first_item = first_item_obj.get('value', '')
            
            # Mimic original "A or B": if val_html_type is truthy, use it, else use val_first_item
            # This means if val_html_type is None or "", val_first_item will be used.
            parsed_content_from_list = val_html_type if val_html_type else val_first_item
        
        if parsed_content_from_list: # If content was found and is truthy from the list
            html_body = parsed_content_from_list
        else:
            # Attempt 2: From entry.summary_detail (if not found or empty from entry.content)
            summary_detail_obj = entry.get("summary_detail")
            val_summary_detail = None # Initialize
            if summary_detail_obj and isinstance(summary_detail_obj, dict):
                val_summary_detail = summary_detail_obj.get('value') # .get defaults to None, or returns actual value

            if val_summary_detail: # If val_summary_detail is truthy (not None, not empty string)
                html_body = val_summary_detail
            else:
                # Attempt 3: From entry.summary (if not found or empty from above)
                summary_val_direct = entry.get("summary") # This is typically the string itself or None
                # Ensure summary_val_direct is a string and truthy, as it might be other types from feedparser
                if summary_val_direct and isinstance(summary_val_direct, str): 
                    html_body = summary_val_direct
        
        # Ensure it's a string
        html_body = str(html_body) if html_body else ""

        # --- Extract Audio URL ---
        audio_url = extract_audio_url(entry)

        # --- Extract Image URL ---
        image_url = None
        if entry.get("image") and isinstance(entry.image, dict) and entry.image.get("href"):
            image_url = entry.image.get("href")
        elif entry.get("media_content"):
             for media in entry.media_content:
                 if isinstance(media, dict) and media.get("medium") == "image" and media.get("url"):
                     image_url = media.get("url")
                     break
        elif entry.get("links"):
            for lnk in entry.links:
                if isinstance(lnk, dict) and lnk.get('rel') == 'enclosure' and lnk.get('type', '').startswith('image/') and lnk.get('href'):
                    image_url = lnk.get('href')
                    break

        # --- Extract Authors ---
        author_name = entry.get("author", source_name) # Default to source name
        authors_list = [name.strip() for name in re.split(r'\s+(?:,|&|and)\s+', author_name)] if author_name else [source_name]

        # --- Extract Tags ---
        tag_names = []
        if "tags" in entry and isinstance(entry.tags, list):
            tag_names = [t.term for t in entry.tags if t and hasattr(t, 'term') and t.term]

        # --- Yield Parsed Data ---
        yield {
            "post_id": post_id,
            "title": title,
            "link": link,
            "published_parsed": entry.get("published_parsed"), # Keep as struct_time for now
            "html_body": html_body,
            "audio_url": audio_url, # Still yield audio_url
            "image_url": image_url,
            "authors_list": authors_list,
            "tags": tag_names,
            "source_name": source_name,
            # Add other potentially useful raw fields if needed
            "raw_entry": entry # Optional: include for debugging or future use
        }

# ================================================================
#                     Main Processing Logic
# ================================================================

# Helper to make entry data JSON serializable
def make_serializable(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, time.struct_time):
        # Convert struct_time to a datetime object then to ISO format
        try:
            dt = struct_time_to_dt(obj) # Use existing helper
            return dt.isoformat() if dt else None
        except:
            return None # Or represent as string if conversion fails
    # Handle feedparser's FeedParserDict by converting to regular dict
    if isinstance(obj, feedparser.FeedParserDict):
         return dict(obj)
    # Recursively handle lists and dictionaries
    if isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    # Add handling for other non-serializable types if encountered
    # For example, if raw_entry contains complex objects:
    # if isinstance(obj, YourCustomClass):
    #     return obj.to_dict() # Assuming a method exists
    try:
        # Attempt default serialization for basic types
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        # Fallback for types json doesn't know how to handle
        return repr(obj) # Represent as string

def main():
    start_time = time.time()
    print("================================================================")
    print(f"Starting Podcast Feed Ingestion Script at {datetime.now(timezone.utc)}")
    print("*** NOTE: Google Transcription code has been REMOVED. Using HTML show notes only. ***")
    print("================================================================")
    # MODIFIED: Add a comment about the performance trade-off
    print("INFO: This version inserts rows individually, which may be slower than batching.")
    logging.info("Running in single-row insert mode (potentially slower than batching).")

    # Directory to save episode samples
    SAMPLE_DIR = "episode_samples"
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    print(f"Saving sample episode data to '{SAMPLE_DIR}/' directory.")

    conn = None
    existing_title_norms: Set[str] = set()
    already_skipped: Set[str] = set()
    total_processed_count = 0
    total_skipped_count = 0
    total_inserted_count = 0
    total_failed_analysis_count = 0
    # MODIFIED: Renamed counter for clarity
    total_db_insert_failures = 0

    try:
        # -------- 1. Database Connection & Setup --------
        print("\n--- Connecting to Database ---")
        logging.info(f"Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False # Manual transaction control
        register_vector(conn) # <-- ADDED for pgvector
        print("Database connection successful.")
        logging.info("Database connection successful.")

        # -------- 2. Fetch Existing Titles --------
        with conn.cursor() as cur:
            print("Fetching existing normalized titles from database...")
            cur.execute("SELECT title_norm FROM content WHERE title_norm IS NOT NULL") # Ensure not null
            existing_title_norms = {row[0] for row in cur.fetchall() if row[0]} # Filter None just in case
            print(f"--> Found {len(existing_title_norms):,} existing titles in the database.")
            logging.info(f"Fetched {len(existing_title_norms)} existing titles.")

            print("Fetching already skipped titles from database...")
            cur.execute("SELECT title_norm FROM skipped_posts WHERE title_norm IS NOT NULL") # Ensure not null
            already_skipped = {row[0] for row in cur.fetchall() if row[0]}
            print(f"--> Found {len(already_skipped):,} already skipped titles in the database.")
            logging.info(f"Fetched {len(already_skipped)} already skipped titles.")

        # -------- 3. Process Feeds --------
        print(f"\n--- Starting Processing for {len(SOURCE_FEEDS)} Feeds ---")
        feed_counter = 0
        for feed_url, source_name in SOURCE_FEEDS.items():
            feed_counter += 1
            print(f"\n===== [{feed_counter}/{len(SOURCE_FEEDS)}] Processing Feed: {source_name} =====")
            posts_in_feed_count = 0
            posts_processed_in_feed = 0
            posts_skipped_in_feed = 0
            sample_saved_for_feed = False # Flag to save only one sample per feed
            successful_insertions_in_feed = 0 # Counter for successful DB inserts this feed

            # Use the simplified iterator
            for entry_data in iter_rss_feed(feed_url, source_name):
                # MODIFIED: Moved sample saving to the very beginning for the first entry
                if not sample_saved_for_feed:
                    try:
                        title_for_log = entry_data.get('title', '[No Title]')[:50]
                        # Sanitize source_name for filename
                        safe_source_name = re.sub(r'[^\w\-]+', '_', source_name)
                        sample_filename = os.path.join(SAMPLE_DIR, f"{safe_source_name}_sample.json")

                        # Make a copy and serialize complex types
                        serializable_data = make_serializable(entry_data.copy())

                        with open(sample_filename, 'w', encoding='utf-8') as f:
                            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
                        logging.info(f"  -> Saved first entry sample for '{title_for_log}...' to {sample_filename}")
                        sample_saved_for_feed = True # Mark as saved for this feed
                    except Exception as sample_err:
                        logging.error(f"  ERROR: Failed to save first entry sample for '{title_for_log}...': {sample_err}", exc_info=True)
                    # --- End Save Sample Data ---

                posts_in_feed_count += 1
                # --- Safety Break (Check before processing) ---
                if posts_in_feed_count > MAX_POSTS_PER_FEED:
                     logging.warning(f"SAFETY BREAK: Reached max check limit ({MAX_POSTS_PER_FEED}) for feed '{source_name}' without reaching target insertions ({TARGET_INSERTIONS_PER_FEED}). Moving to next feed.")
                     break # Stop processing this feed

                total_processed_count += 1
                title = entry_data['title']
                link = entry_data['link']
                post_id = entry_data['post_id'] # For logging
                logging.info(f"--- Processing Entry #{posts_in_feed_count}: '{title[:70]}...' (ID: {post_id}) ---")

                # --- 3a. Normalize Title & Check Existence ---
                title_norm = normalise_title(title)
                # --- DEBUG: Print title and normalized title ---
                # print(f"  DEBUG CHECK: Original Title = '{title}'")
                # print(f"  DEBUG CHECK: Normalized Title = '{title_norm}'")
                # --- End DEBUG ---

                # Check if already processed or known to be skipped
                if title_norm in existing_title_norms or title_norm in already_skipped:
                    logging.debug(f"  SKIP (silent): Normalized title '{title_norm[:70]}...' already in content or skipped_posts cache.")
                    # No counter increment for this type of skip, as it's a pre-existing state.
                    # posts_skipped_in_feed might still be relevant if we want to track how many were skipped *per feed* due to cache.
                    # For now, keeping it truly silent as per instruction for global counters.
                    continue

                if not title_norm:
                    logging.warning(f"  SKIP: Could not normalize title for post ID {post_id}. Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm)
                        logging.debug(f"  Recorded skip for '{title_norm[:70]}...' (empty title_norm) and added to already_skipped cache.")
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for post ID {post_id} (empty title_norm). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    posts_skipped_in_feed += 1
                    continue

                # --- 3b. Parse Date & Check Cutoff ---
                published_dt = struct_time_to_dt(entry_data.get("published_parsed"))
                if not published_dt:
                    logging.warning(f"  SKIP: Could not parse publication date for post ID {post_id}. Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm)
                        logging.debug(f"  Recorded skip for '{title_norm[:70]}...' (invalid date) and added to already_skipped cache.")
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for post '{title_norm[:70]}...' (invalid date). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    posts_skipped_in_feed += 1
                    continue
                if published_dt < CUTOFF_DATE:
                    logging.info(f"  SKIP: Post published ({published_dt.date()}) before cutoff date ({CUTOFF_DATE.date()}). Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm)
                        logging.debug(f"  Recorded skip for '{title_norm[:70]}...' (before cutoff) and added to already_skipped cache.")
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for post '{title_norm[:70]}...' (before cutoff). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    posts_skipped_in_feed += 1
                    continue

                # --- 3c. AI Safety Guardrail ---
                html_body = entry_data.get('html_body', '')
                # MODIFIED: Skip guardrail if testing with MAX_POSTS_PER_FEED = 1
                if not is_ai_safety_post(title, html_body):
                    logging.info(f"  SKIP: Post '{title[:70]}...' failed AI safety guardrail check. Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm)
                        logging.debug(f"  Recorded skip for '{title_norm[:70]}...' (guardrail fail) and added to already_skipped cache.")
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for post '{title_norm[:70]}...' (guardrail fail). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    posts_skipped_in_feed += 1
                    continue

                # --- If passed checks, proceed with full processing ---
                posts_processed_in_feed += 1
                analysis_step_failed = False # Track if any AI step fails for this post

                # --- Initialize content variables ---
                # MODIFIED: Clearer variable names
                transcript_text = None
                html_show_notes = html_body # Keep original HTML if needed
                markdown_from_notes = ""
                text_from_notes = ""
                final_analysis_content = "" # This will hold the content used for AI analysis
                final_full_content_raw = "" # This will be saved to DB 'full_content'
                final_full_content_markdown = "" # This will be saved to DB 'full_content_markdown'
                audio_url = entry_data.get('audio_url') # Get audio URL

                # --- 3d. Attempt Transcription with AssemblyAI ---
                # Only attempt if the client was configured and we have a URL
                if assemblyai_configured and audio_url:
                    transcript_text = transcribe_audio_assemblyai(audio_url, title)
                elif not audio_url:
                     logging.info("  Skipping transcription: No audio URL found in feed entry.")
                # else: # assemblyai_configured is False
                #    logging.info("  Skipping transcription: AssemblyAI client not configured.")

                # --- 3e. Prepare Content for Analysis (Always use HTML now) ---
                if transcript_text:
                    logging.info("  Using AssemblyAI transcript for content analysis.")
                    final_analysis_content = transcript_text
                    final_full_content_raw = transcript_text
                    # Use the raw transcript as markdown for now.
                    # Could potentially add markdown formatting later if needed.
                    final_full_content_markdown = transcript_text
                elif html_show_notes:
                    logging.info("  Transcription failed or skipped. Falling back to HTML show notes for analysis.")
                    try:
                        soup = BeautifulSoup(html_show_notes, 'html.parser')
                        # Existing HTML cleaning logic
                        for element in soup(["script", "style", "iframe", "form", "button", "input", "noscript", "header", "footer", "nav", "aside"]):
                            element.decompose()
                        cleaned_html = str(soup)
                        logging.debug("HTML cleaning successful.")
                        try:
                            # Use markdownify result for both markdown and raw text
                            md_from_html = markdownify(cleaned_html, heading_style="ATX", bullets="-", strip=['script', 'style'], escape_underscores=False)                             
                            markdown_from_notes = md_from_html
                            # Extract plain text from markdown/html for raw content
                            soup_text = BeautifulSoup(md_from_html, 'html.parser') # Parse the markdown
                            text_from_notes = soup_text.get_text(separator=' ', strip=True)

                            final_analysis_content = text_from_notes # Use text from notes for analysis
                            final_full_content_raw = text_from_notes
                            final_full_content_markdown = markdown_from_notes
                            logging.debug("Markdown conversion and text extraction from HTML successful.")
                        except Exception as md_e:
                            logging.error(f"  ERROR: Markdownify conversion/text extraction failed: {md_e}", exc_info=True)
                            # No usable content from HTML if markdownify fails badly
                    except Exception as bs_e:
                        logging.error(f"  ERROR: BeautifulSoup cleaning failed: {bs_e}", exc_info=True)
                        # No usable content from HTML if cleaning fails
                else:
                    logging.warning(f"  No transcript and no HTML show notes found for post ID {post_id}. Cannot perform analysis.")
                    # Ensure content variables are empty/None
                    final_analysis_content = ""
                    final_full_content_raw = ""
                    final_full_content_markdown = ""

                # --- Check for usable content before AI analysis ---
                analysis_content = final_analysis_content # This is from line 1041 in original file
                if not (analysis_content and analysis_content.strip()):
                    logging.warning(f"  SKIP: Post '{title[:70]}...' has no usable content (transcript/HTML). Recording skip.")
                    try:
                        with conn.cursor() as skip_cur:
                            record_skip(skip_cur, post_id, title_norm, link)
                        conn.commit()
                        already_skipped.add(title_norm)
                        logging.debug(f"  Recorded skip for '{title_norm[:70]}...' (no usable content) and added to already_skipped cache.")
                    except (psycopg2.DatabaseError, Psycopg2OpError) as db_err_skip:
                        logging.error(f"  DB ERROR: Failed to record skip for post '{title_norm[:70]}...' (no usable content). Error: {db_err_skip}", exc_info=False)
                        if conn: conn.rollback()
                    total_skipped_count += 1
                    posts_skipped_in_feed += 1
                    analysis_step_failed = True # Also mark as analysis failed for consistency, though we skip insertion.
                    continue # Skip AI analysis and DB insertion

                # --- 3f. Perform AI Analyses (using processed show notes) ---
                sentence_summary, paragraph_summary, key_implication = None, None, None
                db_cluster, db_tags = None, None
                embedding_short_vector, embedding_full_vector = None, None

                # Use the raw text content extracted from the show notes
                # analysis_content = final_analysis_content # Moved up for the skip check

                if analysis_content and analysis_content.strip(): # This check is now somewhat redundant due to the new skip.
                    logging.info("  -> Performing AI analyses on available content...")
                    # OpenAI GPT-4o Analyses
                    sentence_summary = summarize_text(analysis_content)
                    if sentence_summary is None or sentence_summary == "Content was empty.":
                        logging.warning(f"     Sentence summary failed/skipped: {sentence_summary}")
                        sentence_summary = None; analysis_step_failed = True
                    else: logging.debug("     Sentence summary generated.")

                    paragraph_summary = generate_paragraph_summary(analysis_content)
                    if paragraph_summary is None or paragraph_summary == "Content was empty.":
                        logging.warning(f"     Paragraph summary failed/skipped: {paragraph_summary}")
                        paragraph_summary = None; analysis_step_failed = True
                    else: logging.debug("     Paragraph summary generated.")

                    key_implication = generate_key_implication(analysis_content)
                    if key_implication is None or key_implication == "Content was empty.":
                        logging.warning(f"     Key implication failed/skipped: {key_implication}")
                        key_implication = None; analysis_step_failed = True
                    else: logging.debug("     Key implication generated.")

                    original_tags = entry_data.get('tags', [])
                    # Use raw content (from show notes) for tagging
                    cluster_info = generate_cluster_tag(title, original_tags, analysis_content)
                    if isinstance(cluster_info, dict) and "error" in cluster_info:
                         logging.warning(f"     Cluster/tag generation failed/skipped: {cluster_info['error']}")
                         analysis_step_failed = True
                    elif isinstance(cluster_info, dict):
                         db_cluster = cluster_info.get("cluster")
                         db_tags = cluster_info.get("tags")
                         logging.debug(f"     Cluster='{db_cluster}', Tags={db_tags}")
                    else: # Should not happen if helper validation is correct
                         logging.warning(f"     Cluster/tag generation returned unexpected result: {cluster_info}")
                         analysis_step_failed = True

                    # OpenAI Embeddings
                    logging.info("  -> Generating Embeddings...")
                    # Format embeddings to match the standard format
                    short_text_for_embedding = title or ""
                    
                    # Convert topics list to string for embedding
                    topics_str = ""
                    if db_tags:
                        topics_str = ", ".join(db_tags) if isinstance(db_tags, list) else str(db_tags)
                    
                    full_text_for_embedding = f"{sentence_summary or ''}\n{paragraph_summary or ''}\n{key_implication or ''}\n{topics_str}"

                    embedding_short_vector, embedding_full_vector = generate_embeddings(
                        short_text=short_text_for_embedding,
                        full_text=full_text_for_embedding
                    )
                    if embedding_short_vector is None and embedding_full_vector is None:
                        logging.warning(f"     Embedding Generation failed or skipped.")
                        # Not necessarily marking analysis_step_failed, depends if embeddings are critical
                    else: logging.debug("     Embeddings generated.")

                else: # This else block should ideally not be hit if content was empty, due to the new skip.
                      # However, keeping it as a fallback or if the definition of "empty" changes.
                    logging.warning(f"  Skipping AI analysis for post ID {post_id}: No usable content from transcript or HTML (this log might be redundant if already skipped).")
                    analysis_step_failed = True

                if analysis_step_failed:
                    total_failed_analysis_count += 1

                # --- 3g. Prepare Data Tuple for Insertion ---
                # Ensure order matches DB_COLS exactly!
                # 'full_content' is raw text from notes, 'full_content_markdown' is markdown from notes.
                data_tuple = (
                    link,                           # source_url
                    title,                          # title
                    entry_data['source_name'],      # source_type
                    entry_data.get('authors_list', ['Unknown']), # authors (list for ARRAY type)
                    published_dt,                   # published_date (datetime or None)
                    db_tags,                        # topics (list[str] or None)
                    None,                           # score (Podcasts don't usually have score) - Use None
                    entry_data.get('image_url'),    # image_url (str or None)
                    sentence_summary,               # sentence_summary (str or None)
                    paragraph_summary,              # paragraph_summary (str or None)
                    key_implication,                # key_implication (str or None)
                    final_full_content_raw or None, # full_content (Transcript or text from notes)
                    final_full_content_markdown or None, # full_content_markdown (Transcript or markdown notes)
                    None,                           # comment_count (Podcasts don't usually have comments) - Use None
                    db_cluster,                     # cluster_tag (str or None)
                    embedding_short_vector,         # embedding_short (list[float] or None)
                    embedding_full_vector,          # embedding_full (list[float] or None)
                    audio_url,                      # audio_url (original URL, kept for reference)
                    # title_norm                      # Removed: Generated by the database
                )

                # --- Data Validation before adding to batch ---
                if len(data_tuple) != NUM_DB_COLS:
                     logging.critical(f"FATAL MISMATCH: Data tuple length ({len(data_tuple)}) != NUM_DB_COLS ({NUM_DB_COLS}) for post '{title[:50]}...'. Skipping insertion. Check DB_COLS definition.")
                     continue # Skip this post

                # ========================================================
                # MODIFIED: Insert this single row immediately
                # ========================================================
                logging.debug(f"Attempting to insert post '{title[:50]}...'")
                try:
                    with conn.cursor() as cur:
                        # INSERT_SQL handles ON CONFLICT DO NOTHING
                        cur.execute(INSERT_SQL, data_tuple)
                        conn.commit() # Commit this single row transaction
                        logging.info(f"  SUCCESS: DB insert processed for '{title[:70]}...' (inserted or ignored by ON CONFLICT).")
                        total_inserted_count += 1 # Count attempts that didn't raise DB error
                        # Add to set to prevent reprocessing duplicates later in this run
                        # even if ON CONFLICT ignored the insert.
                        existing_title_norms.add(title_norm)
                        successful_insertions_in_feed += 1 # Increment feed-specific counter
                except (psycopg2.DatabaseError, Psycopg2OpError) as db_err:
                    logging.error(f"  DB ERROR: Failed to insert post '{title[:70]}...' (ID: {post_id}). Error: {db_err}", exc_info=False) # Keep log concise
                    logging.debug(f"Failed data tuple: {data_tuple}", exc_info=True) # Add full trace in debug
                    conn.rollback() # Rollback the failed transaction for this row
                    total_db_insert_failures += 1
                except Exception as e:
                    logging.error(f"  UNEXPECTED ERROR: Failed to insert post '{title[:70]}...' (ID: {post_id}). Error: {e}", exc_info=True)
                    conn.rollback() # Rollback on unexpected errors
                    total_db_insert_failures += 1
                # ========================================================
                # End of single row insertion block
                # ========================================================

                # --- Check if target insertions reached ---
                if successful_insertions_in_feed >= TARGET_INSERTIONS_PER_FEED:
                    logging.info(f"Reached target insertions ({TARGET_INSERTIONS_PER_FEED}) for feed '{source_name}'. Moving to next feed.")
                    break # Stop processing this feed

            logging.info(f"===== Finished Feed: {source_name} =====")
            logging.info(f"    Entries checked: {posts_in_feed_count}")
            logging.info(f"    Entries processed (passed checks): {posts_processed_in_feed}")
            logging.info(f"    Entries skipped (duplicate/date/guardrail): {posts_skipped_in_feed}")
            time.sleep(1) # Be polite between feeds

    except Psycopg2OpError as e:
        logging.critical(f"FATAL: Database connection failed: {e}", exc_info=True)
        print(f"FATAL ERROR: Database connection failed: {e}")
    except KeyboardInterrupt:
         logging.warning("KeyboardInterrupt received. Attempting graceful shutdown.")
         print("\nKeyboard interrupt detected. Shutting down...")
         if conn: conn.rollback() # Rollback any pending transaction
    except Exception as e:
        logging.critical(f"CRITICAL ERROR initializing API clients (OpenAI): {e}", exc_info=True)
        # Decide if script should exit if *any* client fails
        sys.exit(1) # Exit if essential clients fail

if __name__ == "__main__":
    main()