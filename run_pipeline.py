#!/usr/bin/env python3
"""
run_pipeline.py
───────────────
Run the whole ingestion flow in order, unbuffered, with per-stage logs and a
compact status log:

  1. ea_lw_query.py            EA Forum / LessWrong / Alignment Forum
  2. podcast_query.py          podcasts (Gemini transcription)
  3. substack_query.py         Substack publications
  4. backfill_sentence_summary.py   looped until "nothing to backfill"
  5. backfill_embeddings.py
  6. novelty_analysis.py       looped until "all caught up"

Usage:
  python run_pipeline.py                 # everything
  python run_pipeline.py --stages ea_lw,substack
  python run_pipeline.py --skip podcast

Logs: run_full_<timestamp>_status.log (one line per stage boundary / notable event)
      run_full_<timestamp>_<stage>[_N].log (full output of each stage)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TS = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
STATUS_LOG = os.path.join(HERE, f"run_full_{TS}_status.log")

# Lines worth surfacing in the status log. Chatty patterns are rate-limited below.
NOTABLE = re.compile(
    r"preflight OK|Found \d+ .* posts meeting|Kept \d+ unique|--> Found|\[FEED |Archive fetch complete|"
    r"\[DB\] \d+ skips|Fetch complete|Finished Feed|Reached target|SAFETY BREAK|FEED MISMATCH|Transcript:|"
    r"RETRY LATER|\[RETRY\]|\[FAIL\]|Content filter|Analysis failed|Embedding generation failed|"
    r"Retrying|Traceback|CRITICAL|ERROR|Processing Summary|Ingestion Summary|Total rows processed|"
    r"DB rows affected|Rows inserted|Entries examined|Analysis/embedding failures|Gate unavailable|"
    r"Transcription failed|DB insert failures|Total execution time|Elapsed:|rows need embeddings|"
    r"Done: \d+ updated|nothing to backfill|Fetched \d+ refs|Finished processing|all caught up|Leaving row"
)
CHATTY = re.compile(r"Processing Post:|DB insert OK|SUCCESS: DB insert|Analysis complete:|AI analysis finished|score=")
CHATTY_EVERY = 25

STAGES: list[tuple[str, list[str], str | None, int]] = [
    # name, command, loop-until marker (substring in output), max loops
    ("ea_lw", ["ea_lw_query.py"], None, 1),
    ("podcast", ["podcast_query.py"], None, 1),
    ("substack", ["substack_query.py"], None, 1),
    ("backfill_summary", ["backfill_sentence_summary.py"], "nothing to backfill", 30),
    ("backfill_embeddings", ["backfill_embeddings.py"], None, 1),
    ("novelty", ["novelty_analysis.py"], "all caught up", 60),
]


def status(msg: str) -> None:
    line = f"{dt.datetime.now().strftime('%H:%M:%S')} {msg}"
    with open(STATUS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_stage(name: str, script: list[str], until: str | None, max_loops: int) -> bool:
    for i in range(1, max_loops + 1):
        suffix = f"_{i}" if max_loops > 1 else ""
        log = os.path.join(HERE, f"run_full_{TS}_{name}{suffix}.log")
        status(f"START {name}{suffix} -> {os.path.basename(log)}")
        t0 = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", LOG_LEVEL="INFO")
        chatty_seen = 0
        matched_until = False
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            proc = subprocess.Popen(
                [sys.executable, "-u", *script], cwd=HERE, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                f.flush()
                stripped = line.rstrip()
                if "httpx" in stripped:
                    continue
                if until and until in stripped:
                    matched_until = True
                if CHATTY.search(stripped):
                    chatty_seen += 1
                    if chatty_seen % CHATTY_EVERY == 0:
                        status(f"  [{name}] {chatty_seen} progress lines so far; latest: {stripped[:140]}")
                    continue
                if NOTABLE.search(stripped):
                    status(f"  [{name}] {stripped[:200]}")
            rc = proc.wait()
        status(f"END {name}{suffix} rc={rc} in {time.time() - t0:.0f}s")
        if rc != 0:
            status(f"FAILED {name}{suffix} (rc={rc}); stopping the pipeline.")
            return False
        if until is None or matched_until:
            return True
    status(f"NOTE {name}: reached max loops ({max_loops}) without the completion marker; continuing.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the AI Safety Feed ingestion pipeline end to end.")
    ap.add_argument("--stages", help="Comma-separated subset in order, e.g. ea_lw,substack")
    ap.add_argument("--skip", help="Comma-separated stages to skip")
    args = ap.parse_args()
    wanted = [s.strip() for s in args.stages.split(",")] if args.stages else [s[0] for s in STAGES]
    skip = {s.strip() for s in args.skip.split(",")} if args.skip else set()
    plan = [s for s in STAGES if s[0] in wanted and s[0] not in skip]

    status(f"PIPELINE START stages={[s[0] for s in plan]} python={sys.executable}")
    t0 = time.time()
    ok = True
    for name, script, until, max_loops in plan:
        if not run_stage(name, script, until, max_loops):
            ok = False
            break
    status(f"PIPELINE {'DONE' if ok else 'ABORTED'} in {(time.time() - t0) / 60:.1f} min")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
