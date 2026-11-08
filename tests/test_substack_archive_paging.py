"""
iter_substack_archive_stubs paging. The script connects to the DB and makes an LLM
preflight call at import time, so the function is read from the source with ast instead.
"""

import ast
import json
import logging
import os
import sys
import types
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import iso_to_dt  # noqa: E402

CUTOFF = datetime(2025, 1, 1, tzinfo=timezone.utc)


class _Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def _load(pages):
    """The iterator wired to a fake requests.get that serves pages in order; returns (fn, calls)."""
    with open(os.path.join(ROOT, "substack_query.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "iter_substack_archive_stubs")
    calls = []

    def fake_get(url, params=None, **_):
        calls.append(dict(params))
        return _Resp(pages[len(calls) - 1])

    env = {
        "requests": types.SimpleNamespace(
            get=fake_get,
            exceptions=types.SimpleNamespace(RequestException=OSError),
        ),
        "time": types.SimpleNamespace(sleep=lambda _: None),
        "json": json, "logging": logging, "iso_to_dt": iso_to_dt, "datetime": datetime,
    }
    exec(compile(ast.Module([node], []), "substack_query.py", "exec"), env)
    return env["iter_substack_archive_stubs"], calls


def _post(i, date="2026-11-01T00:00:00.000Z"):
    return {"id": i, "post_date": date}


def test_follows_next_cursor_and_never_sends_offset():
    fn, calls = _load([{"posts": [_post(1)], "nextCursor": "abc"}, {"posts": [_post(2)], "nextCursor": None}])
    assert [s["id"] for s in fn("x.substack.com", CUTOFF, batch=20)] == [1, 2]
    assert calls == [{"sort": "new", "limit": 20}, {"sort": "new", "limit": 20, "cursor": "abc"}]


def test_stops_at_cutoff_without_fetching_another_page():
    fn, calls = _load([{"posts": [_post(1), _post(2, "2024-06-01T00:00:00.000Z"), _post(3)], "nextCursor": "abc"}])
    assert [s["id"] for s in fn("x.substack.com", CUTOFF, batch=20)] == [1]
    assert len(calls) == 1
