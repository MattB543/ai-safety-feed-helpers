# AI Safety Feed: ingestion pipeline

Python scripts that fill the `content` table behind [aisafetyfeed.com](https://aisafetyfeed.com). Each night they scrape new AI-safety posts from EA Forum, LessWrong/Alignment Forum, Substack and podcasts. Every post gets a summary, tags and embeddings from an LLM, a novelty score, a cleaned title and an illustration. The website and API live in the separate frontend and backend repos.

## Stages

`run_pipeline.py` runs these in order, each as its own process:

| Stage | Script | What it does |
|---|---|---|
| `ea_lw` | `ea_lw_query.py` | EA Forum + LessWrong GraphQL (Alignment Forum = LW posts with `af=true`), score/comment thresholds, posts since 2025-01-01 |
| `podcast` | `podcast_query.py` | Podcast RSS feeds, Gemini transcription (`gemini_transcribe.py`), episodes since 2026-01-01, at most 2 inserts per feed per run |
| `substack` | `substack_query.py` | Substack archive API + RSS fallback, posts since 2025-01-01 |
| `backfill_summary` | `backfill_sentence_summary.py` | Fills missing summaries / key implication (Gemini), 10 rows per run, looped |
| `backfill_embeddings` | `backfill_embeddings.py` | Fills missing `embedding_short` / `embedding_full` |
| `novelty` | `novelty_analysis.py` | Novelty score + "what's new" note against nearest earlier posts, 21 rows per run, looped |

The three scrapers run the relevance gate, one structured analysis call (`llm_common.analyze_content`) and embeddings per post, then insert. Before inserting they dedup on `source_url` first and then on `title_norm` (lowercased, whitespace collapsed). Rejected posts go to `skipped_posts` so they aren't re-checked.

`nightly_refresh.py` then runs `rewrite_titles.py --mode titles` (cleaned titles) and `rewrite_titles.py --mode images-auto` (Gemini illustrations uploaded to Cloudinary).

Shared code: `llm_common.py` (Azure OpenAI client, analysis, relevance gate, embeddings) and `common.py` (DB connection, `normalise_title`, `record_skip`, exit codes, poison-row tracking).

### Exit codes and failure handling

- Stage scripts exit `0` when done (or nothing left), `3` when more work remains (`run_pipeline.py` runs them again, up to a max loop count), and anything else on failure. A stage that hits its loop cap with work left is reported `INCOMPLETE`.
- Every stage runs even if an earlier one failed. The enrichment stages always run, since they also pick up older unfinished rows. `run_pipeline.py` exits 1 if any stage wasn't `OK`.
- Each stage has a wall-clock deadline (`STAGE_DEADLINE_MIN`: podcast 90 min, others 45). On expiry the child is killed and reported `TIMEOUT`.
- Poison rows: the summary/embedding/novelty backfills record per-row failures in `logs/retry_state.json` and skip rows that have failed 3 or more times (logged each run). Delete the file, or one stage's entry, to retry them, e.g. after an API outage.
- Podcast transcripts are cached in `cache/transcripts/<sha1 of audio url>.json` (with a `truncated` flag), so a failed analysis or insert doesn't pay for transcription twice. Delete a file to force a fresh transcription.

## Nightly runbook

- **Schedule:** Windows Task Scheduler task "AI Safety Feed nightly refresh" runs `nightly_refresh.py` with `pythonw.exe` at midnight on Matt's laptop (wake timers on). Register or update it with `powershell -ExecutionPolicy Bypass -File register_nightly_task.ps1`.
- **What it runs:** `run_pipeline.py`, then titles (batch 500), then images for rows created in the **last 14 days** that still have no image (batch 200, about $0.04 each). A failed or capped night is picked up the next night. Older posts are deliberately not backfilled for now (decision 2026-11-03).
- **Manual run** (from this folder): set `PYTHONIOENCODING=utf-8` first, or Windows consoles choke on non-ASCII titles.

  ```powershell
  $env:PYTHONIOENCODING = "utf-8"
  python nightly_refresh.py                    # the whole nightly job
  python run_pipeline.py --stages ea_lw,novelty  # or a subset (also --skip podcast)
  python rewrite_titles.py --mode images-auto --batch-size 200 --created-since 2026-10-20
  ```

- **Cutoffs:** podcasts only ingest episodes from 2026-01-01 on (`CUTOFF_DATE` in `podcast_query.py`). The older backlog is skipped on purpose. EA/LW/Substack use 2025-01-01.
- **Logs** (all in `logs/`, gitignored): `nightly_<ts>.log` (one line per step) plus `nightly_<ts>_<step>.log`; `run_full_<ts>_status.log` (stage boundaries, notable lines and a per-stage `SUMMARY` at the end) plus `run_full_<ts>_<stage>[_N].log`; and `image_cost_<ts>.csv` (per-image cost estimate). `nightly_refresh.py` deletes `logs/*.log` older than 30 days at the start of each run.

## Setup

```powershell
pip install -r requirements.txt                 # pipeline
pip install -r requirements-analysis.txt        # only for analysis/ scripts
copy .env.example .env                          # then fill in values
```

Environment variables (names from `.env.example`; never commit `.env`):

- `AI_SAFETY_FEED_DB_URL`: Postgres connection string
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_REASONING_EFFORT`: chat (summaries, tags, gate, novelty, titles)
- `OPENAI_API_KEY`: embeddings (`text-embedding-3-small`)
- `GEMINI_API_KEY` (optional `GEMINI_TEXT_MODEL`, `GEMINI_TRANSCRIBE_MODEL`): transcription, summary backfill, images
- `SAM_HARRIS_RSS_UID`: personal uid for the Making Sense subscriber feed. The feed is skipped if this is unset.
- `CLOUDINARY_URL`: image hosting
- `TW_USERNAME`, `TW_PASSWORD`: parked Twitter scraper only

Optional knobs read by the scripts: `PODCAST_MAX_INSERTS_PER_FEED`, `MAX_LLM_INPUT_CHARS`, `MAX_AUDIO_BYTES`, `NOVELTY_MAX_*`, `SUBSTACK_ALWAYS_FETCH_RSS`, `GEMINI_IMAGE_MODEL`, `AZURE_USD_PER_M_INPUT/OUTPUT`, `GEMINI_IMAGE_USD_EACH`.

**This repo is public.** `accounts.txt`, `accounts.db`, `twitter.cookies`, `twitter.har` and `.env` hold live credentials. They are gitignored, so never `git add -A` / `git add .`; add files by path.

## Tests

```powershell
pip install pytest
python -m pytest tests
```

These cover pure helpers only (title normalisation, dates, the Substack non-article title filter, retry state, transcript cache). There are no DB or network calls.

## Folder layout

- Root: the pipeline scripts above, `nightly_refresh.py`, `register_nightly_task.ps1`, and the parked scripts below.
- `analysis/`: one-off analysis and export scripts (`embedding_analysis.py`, `ai_safety_feed_stats.py`, `export_url_stats.py`, `export_tweets_markdown.py`, `cluster_visualization.html`). They need `requirements-analysis.txt`.
- `archive/`: retired one-off and superseded scripts, kept for reference. They aren't maintained.
- `tests/`: pytest checks.
- `claude/`: planning notes.
- `logs/`, `cache/`, `episode_samples/`, `generated_images_*/`: local output, gitignored.

## Parked for a future iteration

- **arXiv** (`arxiv_scraper.py`): still writes to the dead `posts` table (the site reads `content`) and uses plain OpenAI instead of `llm_common`. To revive it, port it to `content` + `llm_common` (analysis, gate, embeddings, `common.normalise_title`) and add it as a stage in `run_pipeline.py`.
- **Twitter** (`twitter_scraper.py`, `classify_twitter_ai_safety.py`, `follow_users.py`, `fetch_following.py`, `setup_twitter_accounts.py`, `twitter_migration.py`, `backfill_rt_qt_urls.py`, `recheck_skipped_tweets.py`, `verify_retweets.py`, `zvi_twitter_accounts_scrape.py`): writes the `tweets` / `twitter_profiles` / classification tables. There is no step that promotes tweets into `content`, so nothing reaches the site. It uses twscrape with the live X credentials in `accounts.txt` / `accounts.db`.

## TODO

- **Unique index on `content.source_url`** (follow-up migration). The scrapers now dedup on `source_url` before `title_norm`, but the DB still only enforces `title_norm` (`ON CONFLICT (title_norm)`), so two distinct posts with the same title still collide. Plan:
  1. Find duplicate URLs: `SELECT source_url, count(*) FROM content WHERE source_url IS NOT NULL GROUP BY 1 HAVING count(*) > 1;` and merge or delete them.
  2. `CREATE UNIQUE INDEX CONCURRENTLY content_source_url_key ON content (source_url);`
  3. Switch the scrapers' `ON CONFLICT` target to `source_url` and decide whether `title_norm` stays unique.
