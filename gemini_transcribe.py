#!/usr/bin/env python3
"""
gemini_transcribe.py
────────────────────
Transcribe podcast audio with Gemini (default model gemini-3.8-flash) through
the Files API.

Flow: download the episode enclosure to a temp file → client.files.upload →
wait until the file is ACTIVE → generate_content(model, [audio, prompt]) →
delete the uploaded file and the temp file.

Result semantics (callers rely on this):
  status "ok"       transcript in .text (may be truncated; see .truncated)
  status "skipped"  transcription not attempted (no key, no/invalid URL, file too big)
                    -> caller may fall back to show notes
  status "failed"   a download/upload/model error -> caller should NOT persist a
                    notes-only row or a skip; leave the episode for the next run

Limits (Gemini API): up to 9.5 hours of audio per prompt at 32 tokens per
second of audio; files up to 2 GB through the Files API. Output is capped by
the model's 65,536-token limit (roughly 45k words).

Environment: GEMINI_API_KEY (required), GEMINI_TRANSCRIBE_MODEL (optional).
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

load_dotenv(override=True)

logger = logging.getLogger("gemini_transcribe")

GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
GEMINI_TRANSCRIBE_MODEL = (os.environ.get("GEMINI_TRANSCRIBE_MODEL") or "gemini-3.8-flash").strip()
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(1_500_000_000)))  # Files API limit is 2 GB
DOWNLOAD_TIMEOUT = (30, 300)             # (connect, read) seconds
DEFAULT_HTTP_TIMEOUT_MS = 2 * 60 * 1000   # upload / get / delete
GENERATE_TIMEOUT_MS = 30 * 60 * 1000      # a 3-hour episode can take several minutes
FILE_PROCESSING_POLL_SEC = 3
FILE_PROCESSING_MAX_SEC = 15 * 60
MAX_OUTPUT_TOKENS = 65536

TRANSCRIPT_PROMPT = (
    "Transcribe this podcast episode verbatim in the language spoken (use English if unsure). "
    "Output only the transcript as plain text: no timestamps, no commentary, no headings, no markdown. "
    "Break the text into paragraphs at natural pauses or topic changes, and start a new paragraph "
    "whenever the speaker changes. Omit advertisements and descriptions of music or sound effects."
)

_MIME_BY_EXT = {
    ".mp3": "audio/mp3",
    ".mpeg": "audio/mpeg",
    ".m4a": "audio/m4a",
    ".aac": "audio/aac",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
    ".aiff": "audio/aiff",
}


@dataclass
class TranscriptionResult:
    status: str                  # "ok" | "skipped" | "failed"
    text: Optional[str] = None
    reason: str = ""
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.text)


_client = None


def gemini_transcription_configured() -> bool:
    return bool(GEMINI_API_KEY)


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": DEFAULT_HTTP_TIMEOUT_MS})
    return _client


def _guess_mime(url: str, content_type: str | None) -> str:
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext in _MIME_BY_EXT:
        return _MIME_BY_EXT[ext]
    if content_type and content_type.split(";")[0].strip().startswith("audio/"):
        return content_type.split(";")[0].strip()
    return "audio/mp3"


def _download_audio(audio_url: str, title: str) -> TranscriptionResult | tuple[str, str]:
    """Stream the enclosure to a temp file. Returns (path, mime) or a skipped/failed result."""
    t0 = time.time()
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio")
    os.makedirs(tmp_dir, exist_ok=True)
    suffix = os.path.splitext(urlparse(audio_url).path)[1].lower() or ".mp3"
    path = None
    try:
        with requests.get(audio_url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True,
                          headers={"User-Agent": "AISafetyFeed/1.0 (+https://aisafetyfeed.com)"}) as r:
            r.raise_for_status()
            if r.status_code == 206:
                return TranscriptionResult("failed", reason="server returned a partial (206) response")
            declared = int(r.headers.get("Content-Length") or 0)
            if declared > MAX_AUDIO_BYTES:
                logger.warning("  Skipping transcription for '%s': audio is %.0f MB (limit %.0f MB).",
                               title[:50], declared / 1e6, MAX_AUDIO_BYTES / 1e6)
                return TranscriptionResult("skipped", reason="audio file too large")
            mime = _guess_mime(r.url or audio_url, r.headers.get("Content-Type"))
            fd, path = tempfile.mkstemp(prefix="episode_", suffix=suffix, dir=tmp_dir)
            written = 0
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if written > MAX_AUDIO_BYTES:
                        raise ValueError(f"audio exceeds {MAX_AUDIO_BYTES} bytes")
        if declared and written != declared:
            raise IOError(f"incomplete download: {written} of {declared} bytes")
        if written == 0:
            raise IOError("empty download")
        logger.info("  Downloaded %.1f MB in %.1fs (%s)", written / 1e6, time.time() - t0, mime)
        return path, mime
    except Exception as e:
        logger.error("  Audio download failed for '%s' (%s): %s", title[:50], audio_url[:80], e)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return TranscriptionResult("failed", reason=f"download failed: {e}")


def _is_retryable(exc: BaseException) -> bool:
    """5xx, 429 and transport-level errors are worth retrying; everything else is not."""
    from google.genai import errors as genai_errors
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429:
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        import httpx
        return isinstance(exc, httpx.TransportError)
    except ImportError:
        return False


_retry_transient = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=10, max=90),
    retry=retry_if_exception(_is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@_retry_transient
def _upload(client, path: str, mime_type: str, title: str):
    return client.files.upload(
        file=path,
        config={"mime_type": mime_type, "display_name": (title or "episode")[:120]},
    )


@_retry_transient
def _get_file(client, name: str):
    return client.files.get(name=name)


@_retry_transient
def _generate_transcript(client, uploaded_file):
    from google.genai import types
    return client.models.generate_content(
        model=GEMINI_TRANSCRIBE_MODEL,
        contents=[
            types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type),
            TRANSCRIPT_PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            http_options=types.HttpOptions(timeout=GENERATE_TIMEOUT_MS),
        ),
    )


def _state_name(f) -> str:
    return str(getattr(f.state, "name", f.state)).upper()


def transcribe_local_file(path: str, mime_type: str, title: str) -> TranscriptionResult:
    """Upload a local audio file to Gemini and return a TranscriptionResult."""
    if not GEMINI_API_KEY:
        return TranscriptionResult("skipped", reason="GEMINI_API_KEY not set")
    client = _get_client()
    uploaded = None
    t0 = time.monotonic()
    try:
        uploaded = _upload(client, path, mime_type, title)
        # Audio is usually ACTIVE immediately, but poll (with a real deadline) to be safe.
        deadline = time.monotonic() + FILE_PROCESSING_MAX_SEC
        while _state_name(uploaded) == "PROCESSING":
            if time.monotonic() > deadline:
                return TranscriptionResult("failed", reason="Gemini file processing timed out")
            time.sleep(FILE_PROCESSING_POLL_SEC)
            uploaded = _get_file(client, uploaded.name)
        state = _state_name(uploaded)
        if state != "ACTIVE":
            return TranscriptionResult("failed", reason=f"Gemini file state is {state}")
        logger.info("  Uploaded to Gemini in %.1fs; generating transcript with %s...",
                    time.monotonic() - t0, GEMINI_TRANSCRIBE_MODEL)

        t1 = time.monotonic()
        response = _generate_transcript(client, uploaded)
        text = (response.text or "").strip()
        finish = ""
        try:
            finish = str(response.candidates[0].finish_reason)
        except Exception:
            pass
        usage = getattr(response, "usage_metadata", None)
        logger.info("  Transcript: %d chars in %.1fs (finish=%s, prompt_tokens=%s, output_tokens=%s)",
                    len(text), time.monotonic() - t1, finish,
                    getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None))
        truncated = "MAX_TOKENS" in finish.upper()
        if truncated:
            logger.warning("  Transcript for '%s' hit the output-token limit and is truncated.", title[:50])
        if not text:
            # Distinguish a deterministic block (safety / prohibited content / recitation),
            # where retrying is pointless and the caller should fall back to show notes,
            # from an unexplained empty response, which is worth retrying next run.
            pf = getattr(response, "prompt_feedback", None)
            block_reason = str(getattr(pf, "block_reason", "") or "") if pf else ""
            cands = getattr(response, "candidates", None) or []
            cand_finish = str(getattr(cands[0], "finish_reason", "") or "") if cands else "no candidates"
            reason = f"empty transcript (block_reason={block_reason or 'none'}, candidate finish={cand_finish})"
            deterministic = bool(block_reason) or any(
                k in cand_finish.upper() for k in ("SAFETY", "RECITATION", "PROHIBITED", "BLOCKLIST", "SPII")
            )
            if deterministic:
                logger.warning("  Gemini declined to transcribe '%s': %s. Falling back to show notes.", title[:50], reason)
                return TranscriptionResult("skipped", reason=reason)
            logger.error("  Gemini returned an empty transcript for '%s': %s", title[:50], reason)
            return TranscriptionResult("failed", reason=reason)
        return TranscriptionResult("ok", text=text, truncated=truncated)
    except Exception as e:
        logger.error("  Gemini transcription failed for '%s': %s: %s", title[:50], type(e).__name__, e)
        return TranscriptionResult("failed", reason=f"{type(e).__name__}: {e}")
    finally:
        if uploaded is not None and getattr(uploaded, "name", None):
            try:
                client.files.delete(name=uploaded.name)
            except Exception as e:
                logger.debug("  Could not delete uploaded Gemini file %s: %s", uploaded.name, e)


def transcribe_audio_gemini(audio_url: str, title: str) -> TranscriptionResult:
    """Download an episode's audio and transcribe it with Gemini."""
    if not GEMINI_API_KEY:
        return TranscriptionResult("skipped", reason="GEMINI_API_KEY not set")
    if not audio_url or urlparse(audio_url).scheme not in ("http", "https"):
        logger.warning("  Skipping transcription for '%s': invalid audio URL (%s).", title[:50], str(audio_url)[:60])
        return TranscriptionResult("skipped", reason="invalid audio URL")

    logger.info("  Transcribing '%s' with Gemini (URL: %s...)", title[:50], audio_url[:60])
    downloaded = _download_audio(audio_url, title)
    if isinstance(downloaded, TranscriptionResult):
        return downloaded
    path, mime = downloaded
    try:
        return transcribe_local_file(path, mime, title)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
