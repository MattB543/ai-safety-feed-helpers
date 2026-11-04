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
