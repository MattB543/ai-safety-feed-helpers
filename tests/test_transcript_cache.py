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


def test_malformed_cache_files_are_misses(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(gt, "TRANSCRIPT_CACHE_DIR", str(tmp_path))
    url = "https://example.com/bad.mp3"
    for bad in ({"text": ["a", "list"]}, {"text": "ok", "truncated": "yes"}, ["not", "a", "dict"]):
        with open(gt._cache_path(url), "w", encoding="utf-8") as f:
            json.dump(bad, f)
        assert gt.load_cached_transcript(url) is None
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []
