#!/usr/bin/env python3
"""
llm_common.py
─────────────
Shared LLM helpers for the AI Safety Feed scrapers (EA/LW, Substack, podcasts)
and the backfill scripts.

What lives here:
  • One Azure OpenAI chat client (deployment from AZURE_OPENAI_DEPLOYMENT) with
    retry/backoff on transient errors and a startup preflight that aborts the
    run on configuration problems (bad deployment name, bad key, bad endpoint).
  • analyze_content(): ONE structured-output call that returns the sentence
    summary, paragraph summary, key implication, cluster and tags for a post.
    Previously each scraper made four separate calls, each carrying the full
    post text.
  • is_ai_safety_content(): the yes/no relevance gate. Returns None (not False)
    when the model could not be reached so callers can retry next run instead
    of permanently recording a skip.
  • generate_embeddings(): OpenAI text-embedding-3-small, two vectors per call.
  • build_embedding_text(): the canonical composition of the "full" embedding
    input so scrapers and backfills produce comparable vectors.

Error model (callers rely on this):
  LLMConfigError     the deployment/key/endpoint is wrong -> abort the whole run
  LLMContentFiltered Azure's content filter blocked this post -> record a skip
  LLMError           transient failure after retries -> don't insert, retry next run

Environment:
  AZURE_OPENAI_API_KEY (or AZURE_API_KEY), AZURE_OPENAI_ENDPOINT,
  AZURE_OPENAI_DEPLOYMENT (required, e.g. gpt-5.6-terra),
  AZURE_OPENAI_API_VERSION (default 2024-12-01-preview),
  AZURE_OPENAI_REASONING_EFFORT (default "low"; "none" is sent as the API value
  none; "default" omits the parameter),
  OPENAI_API_KEY (or OPEN_AI_FREE_CREDITS_KEY) for embeddings,
  MAX_LLM_INPUT_CHARS (default 200000).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any, Optional

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AzureOpenAI,
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

load_dotenv(override=True)

logger = logging.getLogger("llm_common")

# ────────────────────────── Configuration ──────────────────────────
AZURE_OPENAI_API_KEY = (os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY") or "").strip()
AZURE_OPENAI_ENDPOINT = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
AZURE_OPENAI_DEPLOYMENT = (os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "").strip()
AZURE_OPENAI_API_VERSION = (os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview").strip()
AZURE_OPENAI_REASONING_EFFORT = (os.environ.get("AZURE_OPENAI_REASONING_EFFORT") or "low").strip().lower()
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_FREE_CREDITS_KEY") or "").strip()

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_LLM_INPUT_CHARS = int(os.environ.get("MAX_LLM_INPUT_CHARS", "200000"))
LLM_REQUEST_TIMEOUT_SEC = 180
ANALYSIS_MAX_COMPLETION_TOKENS = 6000   # includes reasoning tokens on gpt-5 deployments
GATE_MAX_COMPLETION_TOKENS = 1000

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant specialised in AI-safety content analysis."


# ────────────────────────── Exceptions ──────────────────────────
class LLMError(Exception):
    """The model call failed after retries. Safe to retry on a later run."""


class LLMContentFiltered(LLMError):
    """Azure's content filter rejected the input or output. Retrying will not help."""


class LLMConfigError(Exception):
    """Deployment / key / endpoint problem. Not a per-post failure: abort the run."""


def require_env(*, azure: bool = True, embeddings: bool = True) -> None:
    """Exit with a clear message if the LLM configuration is incomplete."""
    missing = []
    if azure and not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY (or AZURE_API_KEY)")
    if azure and not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if azure and not AZURE_OPENAI_DEPLOYMENT:
        missing.append("AZURE_OPENAI_DEPLOYMENT")
    if embeddings and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        msg = "CRITICAL ERROR: missing environment variables: " + ", ".join(missing)
        print(msg, file=sys.stderr)
        logging.critical(msg)
        sys.exit(1)


def preflight(*, embeddings: bool = True) -> None:
    """
    One tiny chat call (and optionally one tiny embedding call) so a wrong
    deployment name, key or endpoint fails the run immediately instead of
    failing every post. Raises LLMConfigError.
    """
    try:
        reply = _chat(
            [{"role": "user", "content": "Reply with the single word OK."}],
            max_completion_tokens=1000,
        )
    except LLMConfigError:
        raise
    except Exception as e:
        raise LLMConfigError(f"Azure OpenAI preflight call failed (deployment '{AZURE_OPENAI_DEPLOYMENT}'): {e}") from e
    logger.info("Azure OpenAI preflight OK: deployment=%s reply=%r", AZURE_OPENAI_DEPLOYMENT, reply[:20])
    if embeddings:
        try:
            _embed(["preflight"])
        except LLMConfigError:
            raise
        except Exception as e:
            raise LLMConfigError(f"OpenAI embeddings preflight failed: {e}") from e
        logger.info("OpenAI embeddings preflight OK: model=%s", EMBEDDING_MODEL)


def check_llm_or_exit(*, embeddings: bool = True) -> None:
    """require_env() + preflight(); exits the process on any configuration problem."""
    require_env(azure=True, embeddings=embeddings)
    try:
        preflight(embeddings=embeddings)
    except LLMConfigError as e:
        msg = f"CRITICAL ERROR: {e}"
        print(msg, file=sys.stderr)
        logging.critical(msg)
        sys.exit(1)


# ────────────────────────── Clients ──────────────────────────
_azure_client: Optional[AzureOpenAI] = None
_openai_client: Optional[OpenAI] = None
_reasoning_effort_supported = True


def get_azure_client() -> AzureOpenAI:
    global _azure_client
    if _azure_client is None:
        if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT):
            raise LLMConfigError("Azure OpenAI is not configured (key/endpoint/deployment missing).")
        # max_retries=0: tenacity below is the single retry layer (SDK retries would multiply it).
        _azure_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            timeout=LLM_REQUEST_TIMEOUT_SEC,
            max_retries=0,
        )
    return _azure_client


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise LLMConfigError("OPENAI_API_KEY is not set; cannot generate embeddings.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=0)
    return _openai_client


# ────────────────────────── Low-level chat call ──────────────────────────
_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_CONFIG_ERRORS = (NotFoundError, AuthenticationError, PermissionDeniedError)


def _extract_text(response) -> str:
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
    return ""


def _is_content_filter_error(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    if code == "content_filter":
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("code") == "content_filter":
            return True
    return "content_filter" in str(exc)


def _reasoning_kwargs() -> dict:
    """AZURE_OPENAI_REASONING_EFFORT: 'default'/'' omits the parameter; anything else
    (low/medium/high/none) is sent verbatim."""
    if not _reasoning_effort_supported or AZURE_OPENAI_REASONING_EFFORT in ("", "default"):
        return {}
    return {"reasoning_effort": AZURE_OPENAI_REASONING_EFFORT}


def _create(client: AzureOpenAI, kwargs: dict):
    """chat.completions.create with error normalisation. BadRequestError that is not a
    content-filter rejection is re-raised for the caller to inspect."""
    try:
        return client.chat.completions.create(**kwargs)
    except _CONFIG_ERRORS as e:
        raise LLMConfigError(
            f"Azure OpenAI configuration problem (deployment '{AZURE_OPENAI_DEPLOYMENT}' at '{AZURE_OPENAI_ENDPOINT}'): {e}"
        ) from e
    except BadRequestError as e:
        if _is_content_filter_error(e):
            raise LLMContentFiltered(f"Azure content filter rejected the request: {e}") from e
        raise


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=5, max=60),
    retry=retry_if_exception_type(_RETRYABLE),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _chat(messages: list[dict], *, max_completion_tokens: int, response_format: dict | None = None) -> str:
    """One chat completion with retries on transient errors. Raises the LLM* exceptions on failure."""
    global _reasoning_effort_supported
    client = get_azure_client()
    # No `temperature`: the gpt-5 family rejects it (and this pipeline only targets gpt-5.x).
    kwargs: dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format
    kwargs.update(_reasoning_kwargs())

    t0 = time.monotonic()
    try:
        response = _create(client, kwargs)
    except BadRequestError as e:
        if "reasoning_effort" in kwargs and "reasoning_effort" in str(e):
            logger.warning("Deployment rejected reasoning_effort=%s; retrying without it.", kwargs["reasoning_effort"])
            _reasoning_effort_supported = False
            kwargs.pop("reasoning_effort", None)
            try:
                response = _create(client, kwargs)
            except BadRequestError as e2:
                raise LLMError(f"Azure OpenAI rejected the request: {e2}") from e2
        else:
            raise LLMError(f"Azure OpenAI rejected the request: {e}") from e

    elapsed = time.monotonic() - t0
    finish_reason = response.choices[0].finish_reason if response.choices else None
    text = _extract_text(response)
    logger.info("Azure OpenAI call finished in %.1fs (finish_reason=%s, %d chars)", elapsed, finish_reason, len(text))

    if finish_reason == "content_filter":
        raise LLMContentFiltered("Azure content filter blocked the model output.")
    if not text:
        raise LLMError(f"Azure OpenAI returned an empty response (finish_reason={finish_reason}).")
    return text


def call_llm_text(prompt: str, *, system: str = DEFAULT_SYSTEM_PROMPT, max_completion_tokens: int = 4000) -> str:
    """Plain-text completion. Raises LLMError / LLMContentFiltered / LLMConfigError."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    try:
        return _chat(messages, max_completion_tokens=max_completion_tokens)
    except (LLMError, LLMConfigError):
        raise
    except APIError as e:
        raise LLMError(f"Azure OpenAI API error: {e}") from e


def truncate_for_llm(text: str | None, limit: int = MAX_LLM_INPUT_CHARS) -> str:
    text = text or ""
    if len(text) > limit:
        logger.info("Truncating LLM input: %d -> %d chars", len(text), limit)
        return text[:limit]
    return text


# ────────────────────────── Taxonomy ──────────────────────────
# Cluster -> list of (canonical tag, synonyms). Canonical tags are stored in
# content.topics; clusters in content.cluster_tag. Keep parentheses out of the
# canonical names (the DB values have always been parentheses-stripped).
TAXONOMY: dict[str, list[tuple[str, str]]] = {
    "Core AI Safety & Alignment": [
        ("AI alignment", "Human alignment"),
        ("Existential risk", "X-risk"),
        ("Threat models", "AI threat models"),
        ("Interpretability", "Interpretability (ML & AI); Transparency"),
        ("Inner alignment", ""),
        ("Outer alignment", ""),
        ("Deceptive alignment", ""),
        ("Eliciting latent knowledge", "ELK"),
        ("Robustness", "Adversarial robustness"),
        ("Alignment field-building", "AI alignment field-building"),
        ("Value learning", "Preference learning; Alignment via human values"),
    ],
    "AI Governance & Policy": [
        ("AI governance", "GovAI"),
        ("Compute governance", "GPU export controls; Chip governance"),
        ("AI regulation", "Regulation"),
        ("Standards & auditing", "Safety standards; Red-teaming"),
        ("Responsible scaling", "Scaling policies; RSF"),
        ("International coordination", "Geopolitics"),
        ("Slowing down AI", "Slow takeoff; Pause AI"),
        ("Open-source models", "Open-source LLMs"),
        ("Policy", "Public policy (generic)"),
        ("Compute controls", "Hardware throttling"),
    ],
    "Technical ML Safety": [
        ("Reinforcement learning", "RL"),
        ("Human feedback", "RLHF; RLAIF"),
        ("Model editing", "Model surgery"),
        ("Scalable oversight", "Debate; Tree-of-thought"),
        ("CoT alignment", "Chain-of-thought alignment"),
        ("Scaling laws", ""),
        ("Benchmarks & evals", ""),
        ("Mechanistic interpretability", ""),
        ("Value decomposition", "Shard theory"),
    ],
    "Forecasting & World Modeling": [
        ("World modeling", ""),
        ("Forecasting", "Quantitative forecasting"),
        ("Prediction markets", ""),
    ],
    "Biorisk & Other GCRs": [
        ("Biorisk", "Biosecurity; Pandemic preparedness"),
        ("Nuclear risk", "Nuclear war; Nuclear winter"),
        ("Global catastrophic risk", "GCR"),
    ],
    "Effective Altruism & Meta": [
        ("Cause prioritization", ""),
        ("Effective giving", ""),
        ("Career choice", "Career planning"),
        ("Community building", "Building effective altruism"),
        ("Field-building", "AI field-building"),
        ("Epistemics & rationality", "Rationality"),
    ],
    "Philosophy & Foundations": [
        ("Decision theory", "CDT; EDT; UDT"),
        ("Moral uncertainty", ""),
        ("Population ethics", ""),
        ("Agent foundations", "Agent foundations research"),
        ("Value drift", ""),
        ("Info hazards", "Information hazards"),
    ],
    "Org-specific updates": [
        ("Anthropic", ""),
        ("OpenAI", ""),
        ("DeepMind", ""),
        ("Meta", ""),
        ("ARC", "Alignment Research Center"),
    ],
}

CLUSTERS: list[str] = list(TAXONOMY.keys())
CANONICAL_TAGS: list[str] = [tag for tags in TAXONOMY.values() for tag, _ in tags]


def _taxonomy_text() -> str:
    lines = ["The format is:", "• Cluster", "- Canonical tag (Synonyms)", ""]
    for cluster, tags in TAXONOMY.items():
        lines.append(f"• {cluster}")
        for tag, syn in tags:
            lines.append(f"- {tag} ({syn})" if syn else f"- {tag}")
        lines.append("")
    return "\n".join(lines).rstrip()


TAXONOMY_TEXT = _taxonomy_text()

ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "sentence_summary": {
            "type": "string",
            "description": "Two concise sentences, maximum 50 words, capturing the core argument or key insight.",
        },
        "paragraph_summary": {
            "type": "string",
            "description": "Markdown: 1-sentence intro, blank line, 3-5 bullets of the form '* **Key concept**: explanation', blank line, 1-sentence conclusion.",
        },
        "key_implication": {
            "type": "string",
            "description": "One sentence, 25-35 words, stating the single most important logical consequence of the content.",
        },
        "cluster": {"type": "string", "enum": CLUSTERS},
        "tags": {
            "type": "array",
            "description": "1 to 4 canonical tags from the taxonomy, most specific first.",
            "items": {"type": "string", "enum": CANONICAL_TAGS},
        },
    },
    "required": ["sentence_summary", "paragraph_summary", "key_implication", "cluster", "tags"],
    "additionalProperties": False,
}

ANALYSIS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "content_analysis", "strict": True, "schema": ANALYSIS_SCHEMA},
}

ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert AI-safety content analyst and taxonomist for an AI-safety news feed. "
    "You read one piece of content and return a JSON object with all requested fields."
)

ANALYSIS_INSTRUCTIONS = """Analyze the AI safety content below and produce ALL of the following fields.

1. sentence_summary: Summarize the content in 2 concise sentences (maximum 50 words). Focus on the core argument, key insight, or main conclusion rather than methodology. Use clear, accessible language while preserving technical accuracy. It should help readers quickly understand what makes this content valuable or interesting and decide whether to read more.

2. paragraph_summary: A structured markdown summary so the reader can quickly understand the main points:
   - A brief 1-sentence introduction highlighting the main point.
   - A blank line, then 3-5 bullet points covering key arguments, evidence, or insights. Format EACH bullet as: * **Key concept or term**: Explanation or elaboration of that point. Keep each bullet to one sentence about one distinct idea, and bold only the key concept at the start.
   - A blank line, then a brief 1-sentence conclusion with the author's recommendation or final thoughts.
   Output only the summary itself (no "Summary:" prefix). Use markdown bolding and italics for readability.

3. key_implication: The single most important logical consequence or implication of the content, in one concise sentence (25-35 words). Focus on what change in thinking, strategy, or priorities follows from accepting the content's conclusions, and on the specific actionable "so what" for an informed AI safety community member. It must be a direct consequence of the argument, not a restatement of the main point.

4. cluster: Exactly one Cluster from the taxonomy below that best captures the main theme.

5. tags: 1 to 4 Canonical Tags from the taxonomy that most precisely describe the content. Prefer the most specific tags that materially help the reader and skip generic or redundant ones. Synonyms in parentheses are hints only; always output the canonical form.
"""


def remove_parentheses_content(text: str | None) -> str | None:
    if not text:
        return text
    cleaned = re.sub(r"\([^)]*\)", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def analyze_content(title: str | None, content_markdown: str | None, source_tags: list[str] | None = None) -> dict:
    """
    Run the single structured analysis call for one post.

    Returns a dict with keys: sentence_summary, paragraph_summary, key_implication,
    cluster_tag, tags (list[str]).

    Raises:
        ValueError            if the content is empty.
        LLMContentFiltered    if Azure's content filter blocked the request (record a skip).
        LLMError              for any other failure after retries (retry on a later run).
        LLMConfigError        if the deployment/key/endpoint is wrong (abort the run).
    """
    if not content_markdown or content_markdown.isspace():
        raise ValueError("Content was empty.")

    title = title or "Untitled"
    tags_text = ", ".join(t for t in (source_tags or []) if t) or "none"
    body = truncate_for_llm(content_markdown)

    prompt = (
        f"{ANALYSIS_INSTRUCTIONS}\n"
        f"--- TAXONOMY ---\n{TAXONOMY_TEXT}\n\n"
        f"--- INPUT ---\n"
        f"Title: {title}\n\n"
        f"Original author-supplied tags (may be noisy or missing): {tags_text}\n\n"
        f"Content (markdown):\n{body}\n"
    )
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = _chat(messages, max_completion_tokens=ANALYSIS_MAX_COMPLETION_TOKENS, response_format=ANALYSIS_RESPONSE_FORMAT)
    except (LLMError, LLMConfigError):
        raise
    except APIError as e:
        raise LLMError(f"Azure OpenAI API error: {e}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Analysis response was not valid JSON: {e}. Text: {raw[:200]}") from e

    if not isinstance(data, dict):
        raise LLMError(f"Analysis response was not a JSON object: {raw[:200]}")

    cluster = remove_parentheses_content(str(data.get("cluster") or "")) or None
    raw_tags = data.get("tags") or []
    tags: list[str] = []
    for t in raw_tags:
        if isinstance(t, str):
            cleaned = remove_parentheses_content(t)
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
    tags = tags[:4]

    result = {
        "sentence_summary": (data.get("sentence_summary") or "").strip() or None,
        "paragraph_summary": (data.get("paragraph_summary") or "").strip() or None,
        "key_implication": (data.get("key_implication") or "").strip() or None,
        "cluster_tag": cluster,
        "tags": tags,
    }
    missing = [k for k in ("sentence_summary", "paragraph_summary", "key_implication", "cluster_tag") if not result[k]]
    if missing or not tags:
        raise LLMError(f"Analysis response is missing fields: {missing + (['tags'] if not tags else [])}")
    return result


# ────────────────────────── Relevance gate ──────────────────────────
GATE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "relevance_gate",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"relevant": {"type": "boolean"}},
            "required": ["relevant"],
            "additionalProperties": False,
        },
    },
}


def is_ai_safety_content(title: str | None, snippet: str | None, *, kind: str = "post") -> Optional[bool]:
    """
    Yes/no guard-rail. Returns True/False from the model, or None if the model
    could not be reached (callers should NOT record a skip in that case).
    LLMConfigError propagates so a bad deployment aborts the run.
    """
    prompt = f"""You are an expert AI-safety content curator deciding what belongs in an AI safety news feed.

Is the following {kind} primarily about AI safety or closely related topics (alignment, AI risk, AI governance, AI policy, technical ML safety, x-risk, interpretability, etc.)?

Rules:
1. The content must substantially discuss AI safety, alignment, governance, or policy.
2. Very general AI/ML technical content or product news is NOT sufficient; there must be a clear safety, risk, ethics, or governance angle.
3. Brief mentions of AI safety in otherwise unrelated content are NOT sufficient.

Title: {(title or '')[:200]}

Content excerpt:
{snippet or ''}
"""
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        raw = _chat(messages, max_completion_tokens=GATE_MAX_COMPLETION_TOKENS, response_format=GATE_RESPONSE_FORMAT)
        data = json.loads(raw)
        return bool(data.get("relevant"))
    except LLMConfigError:
        raise
    except LLMContentFiltered as e:
        # Filtered content can't be analysed anyway: treat as "not for the feed".
        logger.warning("Relevance gate: content filtered (%s); treating as not relevant.", e)
        return False
    except Exception as e:  # LLMError, APIError, JSON errors
        logger.warning("Relevance gate unavailable for '%s': %s", (title or "")[:60], e)
        return None


# ────────────────────────── Embeddings ──────────────────────────
def build_embedding_text(sentence_summary: str | None, paragraph_summary: str | None,
                         key_implication: str | None, tags: list[str] | None) -> str:
    """Canonical text for the 'full' embedding (shared by scrapers and backfills)."""
    topics = ", ".join(t for t in (tags or []) if t)
    return "\n".join([sentence_summary or "", paragraph_summary or "", key_implication or "", topics])


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=3, max=30),
    retry=retry_if_exception_type(_RETRYABLE),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _embed(inputs: list[str]) -> list[list[float]]:
    try:
        response = get_openai_client().embeddings.create(model=EMBEDDING_MODEL, input=inputs)
    except _CONFIG_ERRORS as e:
        raise LLMConfigError(f"OpenAI embeddings configuration problem: {e}") from e
    return [d.embedding for d in response.data]


def generate_embeddings(short_text: str | None, full_text: str | None) -> tuple[list[float] | None, list[float] | None]:
    """
    Embed the short (title) and full (summaries) texts in one request.
    Returns (None, None) if both inputs are empty or the API fails after retries.
    LLMConfigError propagates.
    """
    short_text = (short_text or "").strip()
    full_text = (full_text or "").strip()
    if not short_text and not full_text:
        return None, None
    # The API rejects empty strings, so fall back to the other text if one is empty.
    short_in = short_text or full_text
    full_in = full_text or short_text
    try:
        vectors = _embed([short_in, full_in])
    except LLMConfigError:
        raise
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        return None, None
    if len(vectors) != 2:
        logger.error("Unexpected number of embeddings returned: %d", len(vectors))
        return None, None
    return vectors[0], vectors[1]


def generate_embeddings_batch(short_texts: list[str], full_texts: list[str]) -> list[tuple[list[float] | None, list[float] | None]]:
    """Batch variant for backfills. Empty inputs get (None, None) without an API call. Raises on API failure."""
    if len(short_texts) != len(full_texts):
        raise ValueError("short_texts and full_texts must have the same length")
    payload: list[str] = []
    index: list[tuple[int, int]] = []  # (row, position of the short text in payload; full follows)
    for i, (s, f) in enumerate(zip(short_texts, full_texts)):
        s = (s or "").strip()
        f = (f or "").strip()
        if not s and not f:
            continue
        payload.extend([s or f, f or s])
        index.append((i, len(payload) - 2))
    results: list[tuple[list[float] | None, list[float] | None]] = [(None, None)] * len(short_texts)
    if not payload:
        return results
    vectors = _embed(payload)
    for row, pos in index:
        results[row] = (vectors[pos], vectors[pos + 1])
    return results
