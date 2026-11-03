"""Transcript cache in gemini_transcribe.py (no network: the cache is checked before any download)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemini_transcribe as gt  # noqa: E402


def test_cached_transcript_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "TRANSCRIPT_CACHE_DIR", str(tmp_path))
    url = "https://example.com/episode.mp3"
    assert gt.load_cached_transcript(url) is None

    gt.save_transcript(url, "Episode", gt.TranscriptionResult("ok", text="hello world", truncated=True))
    cached = gt.transcribe_audio_gemini(url, "Episode")
    assert cached.ok and cached.text == "hello world" and cached.truncated and cached.reason == "cached"
