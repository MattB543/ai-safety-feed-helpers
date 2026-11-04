#!/usr/bin/env python3
"""
run_pipeline.py
───────────────
Run the whole ingestion flow in order, unbuffered, with per-stage logs and a
compact status log:

  1. ea_lw_query.py            EA Forum / LessWrong / Alignment Forum
  2. podcast_query.py          podcasts (Gemini transcription)
  3. substack_query.py         Substack publications
  4. backfill_sentence_summary.py   looped while it exits 3 ("more work")
  5. backfill_embeddings.py
  6. novelty_analysis.py       looped while it exits 3

Every stage runs even if an earlier one failed: the scrapers are independent, and
the enrichment stages (4-6) also pick up older unfinished rows, so they always run.
Stage scripts exit 0 = done / nothing left, 3 = more work remains (run again),
anything else = failure (common.EXIT_*). A stage that is still at 3 after its max
loops is INCOMPLETE. A stage that runs past its wall-clock deadline is killed and
reported as TIMEOUT. This script exits 1 if any stage did not finish OK.

Usage:
  python run_pipeline.py                 # everything
  python run_pipeline.py --stages ea_lw,substack
  python run_pipeline.py --skip podcast

Logs (in logs/):
  run_full_<timestamp>_status.log (one line per stage boundary / notable event, and a
                                   per-stage summary at the end)
  run_full_<timestamp>_<stage>[_N].log (full output of each stage)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
TS = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
STATUS_LOG = os.path.join(LOG_DIR, f"run_full_{TS}_status.log")

EXIT_DONE, EXIT_MORE_WORK = 0, 3  # same values as common.EXIT_* (not imported: common loads .env)

# Lines worth surfacing in the status log. Chatty patterns are rate-limited below.
NOTABLE = re.compile(
    r"preflight OK|Found \d+ .* posts meeting|Kept \d+ unique|--> Found|\[FEED |Archive fetch complete|"
    r"\[DB\] \d+ skips|Fetch complete|Finished Feed|Reached target|SAFETY BREAK|FEED MISMATCH|Transcript:|"
    r"RETRY LATER|\[RETRY\]|\[FAIL\]|Content filter|Analysis failed|Embedding generation failed|"
    r"Retrying|Traceback|CRITICAL|ERROR|Processing Summary|Ingestion Summary|Total rows processed|"
    r"DB rows affected|Rows inserted|Entries examined|Analysis/embedding failures|Gate unavailable|"
    r"Transcription failed|DB insert failures|Total execution time|Elapsed:|rows need embeddings|"
    r"Done: \d+ updated|nothing to backfill|Fetched \d+ refs|Finished processing|all caught up|Leaving row|"
    r"failed \d\+ times|truncated|cached transcript|BLOCKED"
)
CHATTY = re.compile(r"Processing Post:|DB insert OK|SUCCESS: DB insert|Analysis complete:|AI analysis finished|score=")
CHATTY_EVERY = 25

STAGES: list[tuple[str, list[str], int]] = [
    # name, command, max loops (looped stages re-run while the script exits 3)
    ("ea_lw", ["ea_lw_query.py"], 1),
    ("podcast", ["podcast_query.py"], 1),
    ("substack", ["substack_query.py"], 1),
    ("backfill_summary", ["backfill_sentence_summary.py"], 30),
    ("backfill_embeddings", ["backfill_embeddings.py"], 1),
    ("novelty", ["novelty_analysis.py"], 60),
]

# Wall-clock budget per stage (all loops together), in minutes. Generous on purpose:
# podcasts normally take 1-45 min, everything else a few minutes.
STAGE_DEADLINE_MIN = {
    "podcast": 90,
    "ea_lw": 45,
    "substack": 45,
    "backfill_summary": 45,
    "backfill_embeddings": 45,
    "novelty": 45,
}


def status(msg: str) -> None:
    line = f"{dt.datetime.now().strftime('%H:%M:%S')} {msg}"
    with open(STATUS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_once(name: str, suffix: str, script: list[str], timeout_sec: float) -> tuple[int, bool]:
    """Run one invocation of a stage script. Returns (exit code, timed_out)."""
    log = os.path.join(LOG_DIR, f"run_full_{TS}_{name}{suffix}.log")
    status(f"START {name}{suffix} -> {os.path.basename(log)}")
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", LOG_LEVEL="INFO")
    chatty_seen = 0
    timed_out = threading.Event()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", *script], cwd=HERE, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )

        def _kill() -> None:
            timed_out.set()
            proc.kill()

        # Reading stdout blocks, so the deadline is enforced from a timer thread.
        timer = threading.Timer(max(timeout_sec, 0), _kill)
        timer.daemon = True
        timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                f.flush()
                stripped = line.rstrip()
                if "httpx" in stripped:
                    continue
                if CHATTY.search(stripped):
                    chatty_seen += 1
                    if chatty_seen % CHATTY_EVERY == 0:
                        status(f"  [{name}] {chatty_seen} progress lines so far; latest: {stripped[:140]}")
                    continue
                if NOTABLE.search(stripped):
                    status(f"  [{name}] {stripped[:200]}")
            rc = proc.wait()
        finally:
            timer.cancel()
    status(f"END {name}{suffix} rc={rc} in {time.time() - t0:.0f}s{' (killed: deadline)' if timed_out.is_set() else ''}")
    return rc, timed_out.is_set()


def run_stage(name: str, script: list[str], max_loops: int) -> str:
    """Run a stage (looping while it exits 3). Returns OK, FAILED, TIMEOUT or INCOMPLETE."""
    deadline = time.time() + STAGE_DEADLINE_MIN.get(name, 45) * 60
    for i in range(1, max_loops + 1):
        suffix = f"_{i}" if max_loops > 1 else ""
        rc, timed_out = run_once(name, suffix, script, deadline - time.time())
        if timed_out:
            status(f"TIMEOUT {name}{suffix}: killed after the {STAGE_DEADLINE_MIN.get(name, 45)} min stage deadline.")
            return "TIMEOUT"
        if rc == EXIT_DONE:
            return "OK"
        if rc != EXIT_MORE_WORK:
            status(f"FAILED {name}{suffix} (rc={rc}); continuing with the next stage.")
            return "FAILED"
        if time.time() >= deadline:
            status(f"INCOMPLETE {name}: stage deadline reached with work remaining.")
            return "INCOMPLETE"
    status(f"INCOMPLETE {name}: reached max loops ({max_loops}) with work remaining.")
    return "INCOMPLETE"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the AI Safety Feed ingestion pipeline end to end.")
    ap.add_argument("--stages", help="Comma-separated subset in order, e.g. ea_lw,substack")
    ap.add_argument("--skip", help="Comma-separated stages to skip")
    args = ap.parse_args()
    wanted = [s.strip() for s in args.stages.split(",")] if args.stages else [s[0] for s in STAGES]
    skip = {s.strip() for s in args.skip.split(",")} if args.skip else set()
    plan = [s for s in STAGES if s[0] in wanted and s[0] not in skip]

    os.makedirs(LOG_DIR, exist_ok=True)
    status(f"PIPELINE START stages={[s[0] for s in plan]} python={sys.executable}")
    t0 = time.time()
    results: list[tuple[str, str, float]] = []
    for name, script, max_loops in plan:
        s0 = time.time()
        results.append((name, run_stage(name, script, max_loops), time.time() - s0))

    ok = all(result == "OK" for _, result, _ in results)
    for name, result, secs in results:
        status(f"SUMMARY {name:<20} {result:<10} {secs / 60:5.1f} min")
    status(f"PIPELINE {'DONE' if ok else 'FINISHED WITH PROBLEMS'} in {(time.time() - t0) / 60:.1f} min")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
