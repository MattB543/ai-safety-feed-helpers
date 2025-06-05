#!/usr/bin/env python3
"""
Ingest "AI Safety" papers from ArXiv, enrich with
Semantic Scholar, summarise with Gemini, embed with OpenAI & Gemini,
and insert into the `posts` table of the AI-Safety-Feed DB.

Environment variables required
------------------------------
ARXIV_QUERY_TERM="ai safety"
ARXIV_MIN_AGE_DAYS=30 (optional - minimum age in days for papers, default 30)
SEMANTIC_SCHOLAR_API_KEY=... (optional - improves rate limits)
OPENAI_API_KEY=...
GEMINI_API_KEY=...
AI_SAFETY_FEED_DB_URL=postgres://...
"""

# Add immediate debug output
print("Script starting - imports beginning...")

# ================================================================
# Imports
# ================================================================
import os, re, json, time, uuid, logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from urllib.parse import urlencode, quote, quote_plus

print("Basic imports completed...")

import feedparser              # ArXiv Atom -> dicts
import requests
import psycopg2
from psycopg2 import extras
from pgvector.psycopg2 import register_vector   # same as EA/LW script
from bs4 import BeautifulSoup
from google import genai as genai
from google.genai import types
from openai import OpenAI, APIError, RateLimitError

print("All imports completed successfully...")

# ================================================================
# Environment
# ================================================================
print("Starting environment setup...")

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env BEFORE using env vars
    print("dotenv loaded successfully")
except ImportError:
    print("dotenv not available (optional)")
    pass  # dotenv is optional

print("Checking environment variables...")

ARXIV_QUERY_TERM         = os.environ.get("ARXIV_QUERY_TERM", "ai safety")
ARXIV_MAX_RESULTS        = 4            # per user request
ARXIV_MIN_AGE_DAYS       = int(os.environ.get("ARXIV_MIN_AGE_DAYS", "30"))  # Default: 30 days
ARXIV_API_URL           = "http://export.arxiv.org/api/query"
S2_API_URL               = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY               = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

OPENAI_API_KEY           = os.environ.get("OPEN_AI_FREE_CREDITS_KEY")
GEMINI_API_KEY           = os.environ.get("GEMINI_API_KEY")
DATABASE_URL             = os.environ.get("AI_SAFETY_FEED_DB_URL")

print(f"Environment variables status:")
print(f"  DATABASE_URL: {'SET' if DATABASE_URL else 'MISSING'}")
print(f"  OPENAI_API_KEY: {'SET' if OPENAI_API_KEY else 'MISSING'}")
print(f"  GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'MISSING'}")
print(f"  S2_API_KEY: {'SET' if S2_API_KEY else 'MISSING'}")

if not all([DATABASE_URL, OPENAI_API_KEY, GEMINI_API_KEY]):
    print("ERROR: Missing required environment variables!")
    print("Required variables:")
    print(f"  AI_SAFETY_FEED_DB_URL: {DATABASE_URL}")
    print(f"  OPEN_AI_FREE_CREDITS_KEY: {OPENAI_API_KEY}")
    print(f"  GEMINI_API_KEY: {GEMINI_API_KEY}")
    raise SystemExit("Missing one or more required API / DB keys")

print("Environment setup completed successfully...")

# ================================================================
# Logging setup
# ================================================================
print("Setting up logging...")

# More focused logging - reduce noise
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Silence noisy libraries
for noisy in ("urllib3", "openai", "httpx", "google.generativeai", "httpcore", "google", "genai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Log S2 API key status
if S2_API_KEY:
    logging.info("Semantic Scholar API key found - using authenticated requests")
else:
    logging.info("No Semantic Scholar API key - using rate-limited public access")

print("Logging setup completed...")

# ================================================================
#                      Database Configuration
# ================================================================
DB_COLS = (
    "uuid", "published_date", "source_updated_at", "title", "title_norm",
    "generated_title", "source_url", "source_type", "authors_display", "authors_ids",
    "content_snippet", "full_content", "short_summary", "long_summary", "key_implication",
    "why_valuable", "image_url", "score", "comment_count",
    "citation_count", "reference_count", "influential_citation_count",
    "doi", "journal_ref", "arxiv_comment",
    "fields_of_study", "venue",
    "first_comment_at", "last_activity_at", "score_timeseries", "comment_timeseries",
    "source_tag_ids", "source_tag_names", "feed_cluster", "feed_tags",
    "reading_time_minutes", "word_count", "external_links", "novelty_score",
    "novelty_note", "embedding_short", "embedding_full", "analysis_version",
    "source_id", "semantic_scholar_id"
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
# Helpers – ArXiv
# ================================================================
def fetch_arxiv_entries(query_term: str, max_results: int = 25, min_age_days: int = 30) -> List[Dict[str, Any]]:
    """
    Query ArXiv API for papers containing `query_term` in title OR abstract,
    excluding papers newer than min_age_days (to ensure citation data has had time to accumulate).
    Returns parsed feedparser dict entries.
    Rate-limited to 1 call every 3 s (ArXiv policy).
    
    Args:
        query_term: Search term to look for in papers
        max_results: Maximum number of results to return
        min_age_days: Minimum age in days for papers (default 30 days)
    """
    # Calculate the cutoff date (papers must be older than this)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=min_age_days)
    # ArXiv date format: YYYYMMDDTTTT where TTTT is HHMM in 24-hour format GMT
    cutoff_str = cutoff_date.strftime("%Y%m%d%H%M")
    
    # Build query with date range - papers submitted before the cutoff date
    # ArXiv date format: [* TO YYYYMMDDTTTT] for papers from beginning of time to cutoff
    search_query = f'all:"{query_term}" AND submittedDate:[* TO {cutoff_str}]'
    logging.info(f"→ ArXiv search: '{query_term}' (excluding papers newer than {min_age_days} days)")
    
    url = f"{ARXIV_API_URL}?search_query={quote_plus(search_query)}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    
    # Add retry logic for connection errors
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Fetch with requests first (supports timeout and proper error handling)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) Gecko/20100101 Firefox/139.0'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the response content with feedparser
            feed = feedparser.parse(response.content)
            
            if feed.bozo:
                logging.error(f"→ Feed parse error: {feed.bozo_exception}")
                if hasattr(feed, 'entries') and len(feed.entries) == 0:
                    logging.error("→ Zero entries returned due to parse error")
                raise RuntimeError(f"ArXiv feed parse error: {feed.bozo_exception}")
            
            logging.info(f"→ Retrieved {len(feed.entries)} entries from ArXiv")
            
            # Honor ArXiv rate-limit: sleep after each call
            time.sleep(3)
            
            return feed.entries
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                logging.warning(f"→ Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error(f"→ All {max_retries} attempts failed. Last error: {e}")
                raise

def arxiv_to_dict(entry) -> Dict[str, Any]:
    """
    Map feedparser entry to a flat dict containing the fields we need
    before enrichment.
    """
    # Strip version suffix (v1, v2, etc.) from ArXiv ID
    raw_id = entry.id if hasattr(entry, 'id') else None
    if not raw_id:
        logging.warning(f"→ Entry has no 'id' attribute: {entry}")
        arxiv_id = None
    else:
        if "/abs/" in raw_id:
            extracted_id = raw_id.split("/abs/")[-1]
        else:
            logging.warning(f"→ Entry ID doesn't contain '/abs/': {raw_id}")
            extracted_id = raw_id
        
        arxiv_id = re.sub(r'v\d+$', '', extracted_id)
    
    authors = []
    if hasattr(entry, 'authors'):
        authors = [a.name.strip() for a in entry.authors]
    else:
        logging.warning(f"→ Entry has no 'authors' attribute")
    
    cats = []
    if hasattr(entry, 'tags'):
        cats = [t["term"] for t in entry.tags] if "tags" in entry else []
    else:
        logging.warning(f"→ Entry has no 'tags' attribute")
    
    links = {}
    if hasattr(entry, 'links'):
        links = { l.rel: l.href for l in entry.links }
    else:
        logging.warning(f"→ Entry has no 'links' attribute")
    
    result = {
        "arxiv_id": arxiv_id,
        "title": entry.title.strip().replace("\n"," ") if hasattr(entry, 'title') else None,
        "summary": entry.summary.strip() if hasattr(entry, 'summary') else None,
        "authors": authors,
        "categories": cats,
        "published": iso_to_dt(entry.published) if hasattr(entry, 'published') else None,
        "updated":   iso_to_dt(entry.updated) if hasattr(entry, 'updated') else None,
        "pdf_url":   links.get("related") or links.get("alternate")+"?format=pdf" if links.get("alternate") else None,
        "arxiv_url": links.get("alternate"),
        "doi": getattr(entry, "arxiv_doi", None),
        "comment": getattr(entry, "arxiv_comment", None),
        "journal_ref": getattr(entry, "arxiv_journal_ref", None),
    }
    
    return result

# ================================================================
# Helpers – Semantic Scholar
# ================================================================
S2_PAPER_FIELDS = "paperId,title,abstract,year,venue,publicationTypes,publicationDate,url,citationCount,influentialCitationCount,referenceCount,isOpenAccess,openAccessPdf,fieldsOfStudy,authors.name,authors.authorId,externalIds"


def enrich_with_semanticscholar(batch: List[str]) -> Dict[str, Any]:
    """
    Call /paper/batch endpoint once for up to 100 ArXiv IDs.
    Returns dict keyed by arxiv_id.
    """
    if not batch:
        logging.warning("Empty batch provided to enrich_with_semanticscholar")
        return {}
    
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    formatted_ids = [f"ARXIV:{pid}" for pid in batch]
    
    # Put fields in query parameters, not JSON body
    params = {'fields': S2_PAPER_FIELDS}
    payload = {"ids": formatted_ids}
    
    logging.info(f"→ Calling Semantic Scholar API for {len(batch)} papers...")
    
    try:
        r = requests.post(f"{S2_API_URL}/paper/batch", 
                          params=params, json=payload, headers=headers, timeout=30)
        
        if r.status_code != 200:
            logging.error(f"Semantic Scholar API error {r.status_code}: {r.text}")
            r.raise_for_status()
            
        response_data = r.json()
        
        # Handle both possible response structures from Semantic Scholar API
        if isinstance(response_data, list):
            # API returned list directly
            papers_list = response_data
        elif isinstance(response_data, dict) and "data" in response_data:
            # API returned dict with "data" key
            papers_list = response_data["data"]
        else:
            logging.warning(f"Unexpected response structure from Semantic Scholar API. Type: {type(response_data)}, Keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'N/A'}")
            return {}
            
        out = {}
        for paper in papers_list:
            if paper is None:  # API returns null for papers not found
                continue
            aid = paper.get("externalIds", {}).get("ArXiv") if paper.get("externalIds") else None
            if aid:
                out[aid] = paper
        
        logging.info(f"→ Successfully enriched {len(out)} papers from Semantic Scholar")
        return out
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error calling Semantic Scholar API: {e}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response from Semantic Scholar API: {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error calling Semantic Scholar API: {e}", exc_info=True)
        return {}

# ================================================================
#                          Gemini Helpers
# ================================================================

def call_gemini_api(prompt: str, model_name: str = "gemini-2.5-flash-preview-04-17") -> str: # Updated to match example model
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
    if not GEMINI_API_KEY:
        logging.warning("call_gemini_api called without GEMINI_API_KEY.")
        return "Analysis skipped (missing API key)."

    try:
        # Use the working client-based approach from the example script
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2) # Fixed: use types instead of gtypes
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

        return result

    except types.generation_types.BlockedPromptException as e: # Fixed: use types instead of gtypes
        logging.error(f"Gemini API call failed due to blocked prompt: {e}")
        return "Error: Analysis blocked due to prompt content."
    except types.generation_types.StopCandidateException as e: # Fixed: use types instead of gtypes
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

def extract_external_links(content: str | None) -> list[str] | None:
    """Extracts all unique external links from HTML content or plain text."""
    if not content:
        return None
    
    links = set()
    
    # First try to parse as HTML
    try:
        soup = BeautifulSoup(content, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Basic check for external links (starts with http/https)
            if href and (href.startswith('http://') or href.startswith('https://')):
                links.add(href)
    except Exception as e:
        logging.debug(f"HTML parsing failed, will try text extraction: {e}")
    
    # Also extract URLs from plain text using regex
    try:
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?)]'
        text_urls = re.findall(url_pattern, content)
        for url in text_urls:
            links.add(url)
    except Exception as e:
        logging.warning(f"Failed to extract URLs from text: {e}")
        
    return sorted(list(links)) if links else None

# ================================================================
# Main
# ================================================================
def main():
    logging.info("=== ArXiv AI-Safety ingestion start ===")

    # 1) Fetch & flatten
    logging.info(f"→ Fetching ArXiv papers for '{ARXIV_QUERY_TERM}' (max {ARXIV_MAX_RESULTS}, min age {ARXIV_MIN_AGE_DAYS} days)")
    raw_entries = fetch_arxiv_entries(ARXIV_QUERY_TERM, ARXIV_MAX_RESULTS, ARXIV_MIN_AGE_DAYS)
    
    if not raw_entries:
        logging.warning("→ No raw entries returned from ArXiv API")
        return
    
    arxiv_posts = []
    for i, entry in enumerate(raw_entries):
        try:
            post_dict = arxiv_to_dict(entry)
            if post_dict.get("arxiv_id"):
                arxiv_posts.append(post_dict)
            else:
                logging.warning(f"→ Entry {i+1} produced no ArXiv ID, skipping")
        except Exception as e:
            logging.error(f"→ Error processing entry {i+1}: {e}", exc_info=True)
    
    arxiv_ids = [p["arxiv_id"] for p in arxiv_posts if p.get("arxiv_id")]
    logging.info(f"→ Processed {len(arxiv_posts)} ArXiv papers: {arxiv_ids}")
    
    if not arxiv_ids:
        logging.error("→ No valid ArXiv IDs extracted from any entries!")
        return

    # 2) Open DB and check for existing papers BEFORE any expensive processing
    logging.info("→ Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    register_vector(conn)
    cur = conn.cursor()

    # Pre-load existing title_norms to avoid processing duplicates
    cur.execute("SELECT title_norm FROM posts")
    existing_titles = {row[0] for row in cur.fetchall()}
    logging.info(f"→ Found {len(existing_titles)} existing papers in database")

    # Filter out papers we already have BEFORE expensive processing
    new_papers = []
    skipped_duplicates = 0
    
    for post in arxiv_posts:
        title_norm = normalise_title(post["title"])
        if title_norm in existing_titles:
            logging.info(f"→ Skipping duplicate: {post['title'][:60]}...")
            skipped_duplicates += 1
            continue
        new_papers.append(post)
    
    logging.info(f"→ Processing {len(new_papers)} new papers ({skipped_duplicates} duplicates skipped)")
    
    if not new_papers:
        logging.info("→ No new papers to process")
        conn.close()
        return

    # Extract ArXiv IDs for new papers only
    new_arxiv_ids = [p["arxiv_id"] for p in new_papers if p.get("arxiv_id")]
    
    # 3) Semantic Scholar enrichment (only for new papers)
    enrichment = enrich_with_semanticscholar(new_arxiv_ids)

    # 4) Initialize OpenAI client once for all papers
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # 5) Process only new papers
    logging.info(f"→ Generating summaries and embeddings for {len(new_papers)} papers...")
    batch = []
    for i, post in enumerate(new_papers):
        logging.info(f"→ Processing paper {i+1}/{len(new_papers)}: {post['title'][:50]}...")
        
        s2 = enrichment.get(post["arxiv_id"], {})
        citation_count = s2.get("citationCount")
        influential_count = s2.get("influentialCitationCount")
        reference_count = s2.get("referenceCount")
        
        # Extract new metadata fields
        fields_of_study = s2.get("fieldsOfStudy")  # list[str] or None
        venue = s2.get("venue")
        
        # Prefer S2's DOI if present, else ArXiv's
        doi = (s2.get("externalIds", {}).get("DOI") 
               if s2 else None) or post.get("doi")
        
        journal_ref = post.get("journal_ref")     # arXiv <arxiv:journal_ref>
        arxiv_comment = post.get("comment")       # arXiv <arxiv:comment>
        
        open_pdf = (s2.get("openAccessPdf", {}).get("url") if s2 else None) or post["pdf_url"]

        # Build textual content for analysis
        abstract_md = post["summary"].replace("\n\n", "\n")
        full_md = f"**Abstract:**\n{abstract_md}"

        # Extract external links from content
        content_for_links = f"{post['title']} {abstract_md}"
        extracted_links = extract_external_links(content_for_links)
        
        # Combine extracted links with PDF URL
        external_links = []
        if open_pdf:
            external_links.append(open_pdf)
        if extracted_links:
            external_links.extend(extracted_links)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_external_links = []
        for link in external_links:
            if link not in seen:
                seen.add(link)
                unique_external_links.append(link)

        # Gemini analyses
        short_sum = generate_short_summary(full_md)
        long_sum  = generate_long_summary(full_md)
        implication = generate_key_implication_text(full_md)
        tags_json   = generate_feed_cluster_and_tags(post["title"], post["categories"], full_md)
        cluster     = tags_json.get("cluster") if isinstance(tags_json, dict) else None
        tag_list    = tags_json.get("tags")    if isinstance(tags_json, dict) else None

        # Embeddings (using the shared client)
        emb_short, emb_full = generate_embeddings(openai_client, post["title"], " ".join([short_sum,long_sum,implication]))

        # Assemble DB tuple (fill unused fields with None)
        uuid_str = str(uuid.uuid4())
        title_norm = normalise_title(post["title"])
        word_count, reading_minutes = calculate_word_count_and_reading_time(abstract_md)
        
        # Ensure proper defaults if word_count is None or 0
        if word_count is None or word_count == 0:
            word_count = 0
            reading_minutes = 0
        elif reading_minutes == 0 and word_count > 0:
            reading_minutes = 1

        # Get Semantic Scholar paper ID
        semantic_scholar_paper_id = s2.get("paperId") if s2 else None

        data = (
            uuid_str,
            post["published"],            # published_date
            post["updated"],              # source_updated_at
            post["title"], title_norm,    # title, title_norm
            post["title"],                # generated_title (use original)
            post["arxiv_url"], "ArXiv",   # url, source_type
            post["authors"],              # authors_display
            [a.get("authorId") for a in s2.get("authors",[])],  # authors_ids (S2 IDs)
            None,                         # content_snippet
            full_md,                      # full_content
            short_sum, long_sum, implication,
            None,                         # why_valuable
            None,                         # image_url
            influential_count,            # score (proxy: influential citations)
            None,                         # comment_count
            citation_count,               # citation_count (dedicated column)
            reference_count,              # reference_count
            influential_count,            # influential_citation_count
            doi,                          # doi
            journal_ref,                  # journal_ref
            arxiv_comment,                # arxiv_comment
            fields_of_study,              # fields_of_study
            venue,                        # venue
            None, None,                   # first_comment_at, last_activity_at
            None, None,                   # score_timeseries, comment_timeseries
            None, post["categories"],     # source_tag_ids, source_tag_names
            cluster, tag_list,            # feed_cluster, feed_tags
            reading_minutes, word_count,
            unique_external_links,        # external_links
            None, None,                   # novelty_score, novelty_note
            emb_short, emb_full,
            "1.0",                        # analysis_version
            post["arxiv_id"],             # source_id (ArXiv ID)
            semantic_scholar_paper_id     # semantic_scholar_id (Semantic Scholar paper ID)
        )
        batch.append(data)

    # 6) Insert
    if batch:
        try:
            logging.info(f"→ Attempting to insert {len(batch)} records into database...")
            
            # Log the first record's key details for debugging
            if batch:
                sample_record = batch[0]
                logging.info(f"→ Sample record: UUID={sample_record[0]}, Title='{sample_record[3][:50]}...', Title_norm='{sample_record[4]}'")
            
            # Execute the batch insert
            extras.execute_batch(cur, INSERT_SQL, batch)
            
            # Check how many rows were actually affected
            if hasattr(cur, 'rowcount'):
                logging.info(f"→ Rows affected by INSERT: {cur.rowcount}")
            
            # Commit the transaction
            logging.info("→ Committing transaction...")
            conn.commit()
            logging.info("→ Transaction committed successfully")
            
            # Verify the insert by checking if the record exists
            sample_title_norm = batch[0][4]  # title_norm is at index 4
            cur.execute("SELECT uuid, title FROM posts WHERE title_norm = %s", (sample_title_norm,))
            verification_result = cur.fetchone()
            
            if verification_result:
                logging.info(f"→ Verification successful: Record found in DB with UUID {verification_result[0]}")
            else:
                logging.error(f"→ Verification failed: Record with title_norm '{sample_title_norm}' not found in DB after insert!")
                
                # Additional debugging - check if there are any constraint violations
                cur.execute("SELECT COUNT(*) FROM posts WHERE title_norm = %s", (sample_title_norm,))
                count_result = cur.fetchone()
                logging.info(f"→ Count check: {count_result[0] if count_result else 'None'} records with this title_norm")
                
                # Check if the INSERT was skipped due to ON CONFLICT
                logging.info("→ Checking if INSERT was skipped due to duplicate title_norm...")
                cur.execute("SELECT uuid, title, published_date FROM posts WHERE title_norm = %s", (sample_title_norm,))
                existing_record = cur.fetchone()
                if existing_record:
                    logging.info(f"→ Found existing record: UUID={existing_record[0]}, Published={existing_record[2]}")
                else:
                    logging.error("→ No existing record found either - this suggests a different issue")
            
            logging.info(f"Successfully processed {len(batch)} new ArXiv posts")
            
        except psycopg2.Error as e:
            logging.error(f"→ Database error during insert: {e}")
            logging.error(f"→ Error code: {e.pgcode}")
            logging.error(f"→ Error details: {e.pgerror}")
            conn.rollback()
            raise
        except Exception as e:
            logging.error(f"→ Unexpected error during insert: {e}", exc_info=True)
            conn.rollback()
            raise
    else:
        logging.info("Nothing new to insert")

    conn.close()
    logging.info("=== Ingestion complete ===")

if __name__ == "__main__":
    main()