"""podcast_query.iter_rss_feed refuses a 'podcast' feed with no audio at all (no network)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import podcast_query as pq  # noqa: E402


class _Resp:
    status_code = 200
    url = "https://site.example/podcast/feed/"
    headers = {"Content-Type": "application/rss+xml; charset=utf-8"}

    def __init__(self, items: str):
        self.content = (b'<?xml version="1.0"?><rss version="2.0"><channel><title>Some Podcast</title>'
                        + items.encode() + b"</channel></rss>")

    def raise_for_status(self):
        pass


WEB_ITEMS = "".join(f"<item><title>Episode {i}</title><link>https://site.example/podcast/ep-{i}/</link>"
                    f"<pubDate>Tue, 27 Oct 2026 16:09:22 +0000</pubDate></item>" for i in range(3))
AUDIO_ITEM = ('<item><title>Episode 9</title><link>https://site.example/podcast/ep-9/</link>'
              '<enclosure url="https://cdn.example/ep9.mp3" type="audio/mpeg" length="1"/></item>')


def _run(monkeypatch, items):
    monkeypatch.setattr(pq.requests, "get", lambda *a, **k: _Resp(items))
    pq.FEED_FETCH_FAILURES.clear()
    return list(pq.iter_rss_feed("https://site.example/podcast/feed/", "Some Podcast"))


def test_feed_with_no_audio_is_a_feed_failure(monkeypatch):
    # Like futureoflife.org/podcast/feed/ (a WordPress post feed): nothing is yielded, so no
    # episode can be recorded as a permanent "no usable content" skip, and the stage fails.
    assert _run(monkeypatch, WEB_ITEMS) == []
    assert pq.FEED_FETCH_FAILURES == ["Some Podcast"]


def test_feed_with_some_audio_is_processed(monkeypatch):
    entries = _run(monkeypatch, WEB_ITEMS + AUDIO_ITEM)
    assert len(entries) == 4
    assert [e["audio_url"] for e in entries if e["audio_url"]] == ["https://cdn.example/ep9.mp3"]
    assert pq.FEED_FETCH_FAILURES == []
