"""Pure helpers in common.py (no DB or network)."""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import iso_to_dt, normalise_title  # noqa: E402


def test_normalise_title_matches_title_norm():
    assert normalise_title("  The   Case\tFor\nAI  Pauses ") == "the case for ai pauses"
    assert normalise_title("") == ""
    assert normalise_title(None) == ""
    # Punctuation is kept: existing content.title_norm values depend on it.
    assert normalise_title("AI #116: Fast & Slow") == "ai #116: fast & slow"


def test_iso_to_dt():
    assert iso_to_dt("2026-11-03T10:00:00Z") == datetime(2026, 11, 3, 10, tzinfo=timezone.utc)
    assert iso_to_dt("2026-11-03T12:00:00+02:00") == datetime(2026, 11, 3, 10, tzinfo=timezone.utc)
    assert iso_to_dt("2026-11-03T10:00:00") == datetime(2026, 11, 3, 10, tzinfo=timezone.utc)
    assert iso_to_dt("not a date") is None
    assert iso_to_dt(None) is None


def test_retry_state_skips_rows_after_three_unfinished_attempts(tmp_path):
    from common import RetryState

    path = str(tmp_path / "retry_state.json")
    for _ in range(3):
        RetryState("novelty", path).begin([7, 8])
    state = RetryState("novelty", path)
    state.begin([9])
    state.succeeded([8])
    state = RetryState("novelty", path)
    assert state.skip_ids() == [7]
    assert state.counts == {"7": 3, "9": 1}
    assert RetryState("backfill_summary", path).skip_ids() == []


def test_retry_state_merges_concurrent_writers(tmp_path):
    from common import RetryState

    path = str(tmp_path / "retry_state.json")
    a = RetryState("novelty", path)
    b = RetryState("backfill_summary", path)
    c = RetryState("novelty", path)
    a.begin([1])
    b.begin([2])
    c.begin([1, 3])  # c loaded before a's write; must not overwrite it
    assert RetryState("novelty", path).counts == {"1": 2, "3": 1}
    assert RetryState("backfill_summary", path).counts == {"2": 1}
    assert not os.path.exists(path + ".lock")
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []


def test_retry_state_ignores_stale_lock(tmp_path):
    import common

    path = str(tmp_path / "retry_state.json")
    with open(path + ".lock", "w"):
        pass
    old = os.path.getmtime(path + ".lock") - common._LOCK_STALE_SEC - 5
    os.utime(path + ".lock", (old, old))
    common.RetryState("novelty", path).begin([4])
    assert common.RetryState("novelty", path).counts == {"4": 1}


def test_retry_state_drops_malformed_entries(tmp_path):
    import json
    from common import RetryState

    path = str(tmp_path / "retry_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"novelty": {"7": "3", "8": 2, "x": 1, "9": True}, "other": [1, 2]}, f)
    state = RetryState("novelty", path)
    assert state.counts == {"8": 2}
    assert state.skip_ids() == []
    with open(path, "w", encoding="utf-8") as f:
        f.write("[1, 2, 3]")
    assert RetryState("novelty", path).counts == {}
