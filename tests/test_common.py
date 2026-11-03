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


def test_retry_state_skips_rows_after_three_failures(tmp_path):
    from common import RetryState

    path = str(tmp_path / "retry_state.json")
    for _ in range(3):
        RetryState("novelty", path).record(failed=[7, 8])
    RetryState("novelty", path).record(failed=[9], succeeded=[8])
    state = RetryState("novelty", path)
    assert state.skip_ids() == [7]
    assert state.counts == {"7": 3, "9": 1}
    assert RetryState("backfill_summary", path).skip_ids() == []
