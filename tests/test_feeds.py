"""Feed helpers in common.py (no network: a fake response object is parsed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import is_duplicate_episode_url, parse_feed_response  # noqa: E402


def test_episode_url_dedup_needs_matching_audio():
    stored = {"https://show.example/": {"https://cdn.example/ep1.mp3", "https://cdn.example/ep0.mp3"},
              "https://show.example/ep/2": {"https://cdn.example/ep2.mp3"}}
    # Show homepage stored for older episodes: a new episode with the same link is NOT a duplicate...
    assert not is_duplicate_episode_url("https://show.example/", "https://cdn.example/ep3.mp3", stored)
    # ...but every older episode sharing that homepage is still recognised.
    assert is_duplicate_episode_url("https://show.example/", "https://cdn.example/ep0.mp3", stored)
    assert is_duplicate_episode_url("https://show.example/", "https://cdn.example/ep1.mp3", stored)
    # Same link and same audio: the same episode (e.g. retitled).
    assert is_duplicate_episode_url("https://show.example/ep/2", "https://cdn.example/ep2.mp3", stored)
    # Unknown link, or no audio URL to compare: fall back to the title check.
    assert not is_duplicate_episode_url("https://show.example/ep/9", "https://cdn.example/ep9.mp3", stored)
    assert not is_duplicate_episode_url("https://show.example/ep/2", None, stored)


class _FakeResponse:
    url = "https://feeds.example.com/show/rss.xml"
    headers = {"Content-Type": "application/rss+xml; charset=utf-8"}
    content = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Show</title>
<item><title>Episode 123</title><link>/episodes/123</link>
<enclosure url="audio/123.mp3" type="audio/mpeg" length="1"/></item></channel></rss>"""


def test_parse_feed_response_resolves_relative_links():
    entry = parse_feed_response(_FakeResponse()).entries[0]
    assert entry.link == "https://feeds.example.com/episodes/123"
    # Enclosures are not resolved by feedparser; podcast_query joins them with resp.url itself.


class _EmptyFeedResponse(_FakeResponse):
    content = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Show</title></channel></rss>"""


class _HtmlResponse(_FakeResponse):
    headers = {"Content-Type": "text/html"}
    content = b"<html><body>Please log in</body></html>"


def test_valid_empty_feed_is_distinguishable_from_html():
    # The scrapers treat "well-formed, no items" as nothing-to-do and only HTML/junk as a failure.
    empty = parse_feed_response(_EmptyFeedResponse())
    assert not empty.entries and not empty.bozo and empty.get("feed")
    html = parse_feed_response(_HtmlResponse())
    assert not html.entries and (html.bozo or not html.get("feed"))


class _Resp:
    def __init__(self, status, ctype, text="", extra=None):
        self.status_code, self.text = status, text
        self.headers = {"Content-Type": ctype, **(extra or {})}


def test_bot_checkpoint_is_not_a_rate_limit():
    from common import is_bot_checkpoint
    page = "<!DOCTYPE html><html><head><title>Vercel Security Checkpoint</title></head></html>"
    assert is_bot_checkpoint(_Resp(429, "text/html; charset=utf-8", page))
    assert is_bot_checkpoint(_Resp(403, "text/plain", "", {"x-vercel-mitigated": "challenge"}))
    # A real rate limit (JSON or plain) still goes through the normal retry path.
    assert not is_bot_checkpoint(_Resp(429, "application/json", '{"error":"rate limited"}'))
    assert not is_bot_checkpoint(_Resp(429, "text/html", "<html>Too Many Requests</html>"))
    assert not is_bot_checkpoint(_Resp(200, "text/html", page))
