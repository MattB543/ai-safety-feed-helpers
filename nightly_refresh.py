#!/usr/bin/env python3
"""
nightly_refresh.py
──────────────────
Bring the live feed fully up to date in one go (what the nightly scheduled task runs):

  1. run_pipeline.py                        scrape + summaries/tags/embeddings + novelty
  2. rewrite_titles.py --mode titles        cleaned titles for every row missing one
  3. rewrite_titles.py --mode images-auto   images for rows created in the last 14 days
                                            that still lack one (capped at IMAGE_BATCH)

Steps 2-3 run even if step 1 fails, so rows it already inserted still get titles/images.
While running it asks Windows not to idle-sleep: after a wake-timer wake the machine
otherwise drops back to sleep after ~2 minutes without user input.

Usage:
  python nightly_refresh.py
  (scheduled via register_nightly_task.ps1, which launches it with pythonw.exe so no window opens)

Logs: logs/nightly_<timestamp>.log (one line per step) and logs/nightly_<timestamp>_<step>.log
      plus run_pipeline.py's own logs/run_full_<timestamp>_* logs. logs/*.log files older
      than LOG_RETENTION_DAYS are deleted at the start of each run.
"""

from __future__ import annotations

import datetime as dt
import glob
import os
import socket
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TS = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(HERE, "logs")
LOG = os.path.join(LOG_DIR, f"nightly_{TS}.log")

# Launched with pythonw.exe by the scheduled task; run the steps with the console python.exe next to it.
PYTHON = os.path.join(os.path.dirname(sys.executable), "python.exe") if os.name == "nt" else sys.executable
TITLE_BATCH = 500
IMAGE_BATCH = 200          # cost guard: ~$0.043 per image
# Images are generated for rows created in this window that still lack one, so a failed or
# capped night is picked up the next night. Older posts are deliberately not backfilled
# for now (Matt's decision, 2026-11-03).
IMAGE_LOOKBACK_DAYS = 14
LOG_RETENTION_DAYS = 30
NETWORK_WAIT_SEC = 300     # Wi-Fi can take a while to come back after a wake


def log(msg: str) -> None:
    line = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if sys.stdout is not None:
        print(line, flush=True)


def keep_awake() -> None:
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def wait_for_network() -> bool:
    deadline = time.time() + NETWORK_WAIT_SEC
    while True:
        try:
            socket.create_connection(("www.lesswrong.com", 443), timeout=10).close()
            return True
        except OSError:
            if time.time() > deadline:
                return False
            time.sleep(10)


def prune_old_logs() -> int:
    """Delete logs/*.log files older than LOG_RETENTION_DAYS. Returns how many were removed."""
    cutoff = time.time() - LOG_RETENTION_DAYS * 86400
    removed = 0
    for path in glob.glob(os.path.join(LOG_DIR, "*.log")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


def run_step(name: str, args: list[str]) -> int:
    step_log = os.path.join(LOG_DIR, f"nightly_{TS}_{name}.log")
    log(f"START {name}: {' '.join(args)} -> {os.path.basename(step_log)}")
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with open(step_log, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([PYTHON, "-u", *args], cwd=HERE, env=env,
                             stdout=f, stderr=subprocess.STDOUT, creationflags=flags)
    log(f"END {name} rc={rc} in {(time.time() - t0) / 60:.1f} min")
    return rc


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    keep_awake()
    pruned = prune_old_logs()
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=IMAGE_LOOKBACK_DAYS)).isoformat(timespec="seconds")
    log(f"NIGHTLY START python={PYTHON} images for rows created since {since}; pruned {pruned} old log(s)")
    if not wait_for_network():
        log(f"NIGHTLY ABORTED: no network after {NETWORK_WAIT_SEC}s")
        return 1

    rcs = [
        run_step("pipeline", ["run_pipeline.py"]),
        run_step("titles", ["rewrite_titles.py", "--mode", "titles", "--batch-size", str(TITLE_BATCH)]),
        run_step("images", ["rewrite_titles.py", "--mode", "images-auto", "--batch-size", str(IMAGE_BATCH),
                            "--created-since", since]),
    ]
    ok = all(rc == 0 for rc in rcs)
    log(f"NIGHTLY {'DONE' if ok else 'FINISHED WITH ERRORS'} rcs={rcs}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # pythonw has no stderr, so make sure a crash lands in the log.
        os.makedirs(LOG_DIR, exist_ok=True)
        log("NIGHTLY CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
