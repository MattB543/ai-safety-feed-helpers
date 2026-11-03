"""
NON_ARTICLE_TITLE_RE in substack_query.py. The script connects to the DB and makes an
LLM preflight call at import time, so the regex is read from the source with ast instead.
"""

import ast
import os
import re

SUBSTACK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "substack_query.py")


def _load_pattern() -> re.Pattern:
    with open(SUBSTACK, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NON_ARTICLE_TITLE_RE" for t in node.targets):
            return eval(compile(ast.Expression(node.value), SUBSTACK, "eval"), {"re": re})
    raise AssertionError("NON_ARTICLE_TITLE_RE not found in substack_query.py")


PATTERN = _load_pattern()


def test_source_has_no_control_characters():
    with open(SUBSTACK, "rb") as f:
        assert b"\x08" not in f.read()


def test_recurring_formats_are_excluded():
    for title in ("Open Thread 412", "Mantic Monday 3/4/26", "Open Hidden Open Thread 412.5",
                  "Highlights From The Comments On AI Pauses", "Links For November 2026"):
        assert PATTERN.search(title), title


def test_normal_titles_pass():
    for title in ("Why I Am Not (As Much Of) A Doomer", "The Case For An Open Threads Policy",
                  "Mantic Mondays are back"):
        assert not PATTERN.search(title), title
