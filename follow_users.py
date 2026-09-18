#!/usr/bin/env python3
"""
follow_users.py
───────────────
Follow Twitter/X accounts from the leaderboard that AISafetyFeed doesn't yet follow.

Uses twscrape's authenticated httpx client to POST to the v1.1 friendships/create
endpoint. Bypasses twscrape's queue system to avoid rate-limit locks from read ops.

Usage:
    python follow_users.py                    # run it
    python follow_users.py --dry-run          # preview only
    python follow_users.py --delay-min 45 --delay-max 90  # slower
    python follow_users.py --limit 50         # only follow first 50
"""

import asyncio
import argparse
import json
import logging
import os
import random
import re
import time

import httpx

# ── Monkey-patch twscrape for Twitter's malformed JSON (Dec 2025+) ──────────
def _script_url(k: str, v: str):
    return f"https://abs.twimg.com/responsive-web/client-web/{k}.{v}.js"

def _patched_get_scripts_list(text: str):
    scripts = text.split('e=>e+"."+')[1].split('[e]+"a.js"')[0]
    try:
        for k, v in json.loads(scripts).items():
            yield _script_url(k, f"{v}a")
    except json.decoder.JSONDecodeError:
        fixed = re.sub(
            r'([,\{])(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*):',
            r'\1\2"\3"\4:',
            scripts,
        )
        for k, v in json.loads(fixed).items():
            yield _script_url(k, f"{v}a")

from twscrape import xclid
xclid.get_scripts_list = _patched_get_scripts_list

from twscrape import API
from twscrape.queue_client import XClIdGenStore
from twscrape.utils import encode_params

# ── Config ──────────────────────────────────────────────────────────────────
TWSCRAPE_DB = "accounts.db"
LEADERBOARD_FILE = "20260304_final_leaderboard_300_x_links.txt"
ACCOUNT_USERNAME = "AISafetyFeed"
ID_CACHE_FILE = "follow_users_id_cache.json"

FOLLOW_URL = "https://x.com/i/api/1.1/friendships/create.json"
FOLLOW_PATH = "/i/api/1.1/friendships/create.json"

# GraphQL endpoint for resolving usernames (bypasses twscrape queue)
OP_USER_BY_SCREEN_NAME = "1VOOyvKkiI3FMmkeDNxM9A/UserByScreenName"
GQL_URL = "https://x.com/i/api/graphql"

FOLLOW_FORM_FIELDS = {
    "include_profile_interstitial_type": "1",
    "include_blocking": "1",
    "include_blocked_by": "1",
    "include_followed_by": "1",
    "include_want_retweets": "1",
    "include_mute_edge": "1",
    "include_can_dm": "1",
    "include_can_media_tag": "1",
    "include_ext_is_blue_verified": "1",
    "include_ext_verified_type": "1",
    "include_ext_profile_image_shape": "1",
    "skip_status": "1",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_target_usernames(filepath: str) -> list[str]:
    """Load usernames from the leaderboard file (one x.com URL per line)."""
    usernames = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "x.com/" in line:
                username = line.split("x.com/")[-1].strip("/")
                if username:
                    usernames.append(username.lower())
            else:
                usernames.append(line.lower())
    return usernames


def load_id_cache() -> dict[str, str]:
    if os.path.exists(ID_CACHE_FILE):
        with open(ID_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_id_cache(cache: dict[str, str]):
    with open(ID_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def make_browser_client(account) -> httpx.AsyncClient:
    """Create an authenticated httpx client that mimics a real browser session."""
    client = account.make_client()
    # Set headers to match what the browser actually sends
    client.headers["content-type"] = "application/x-www-form-urlencoded"
    client.headers["x-twitter-auth-type"] = "OAuth2Session"
    client.headers["origin"] = "https://x.com"
    client.headers["referer"] = "https://x.com/"
    client.headers["accept"] = "*/*"
    client.headers["sec-fetch-dest"] = "empty"
    client.headers["sec-fetch-mode"] = "cors"
    client.headers["sec-fetch-site"] = "same-origin"
    return client


async def resolve_user_id_raw(client, clid_gen, username: str) -> str | None:
    """Resolve a username to user ID using the raw GraphQL endpoint (no twscrape queue)."""
    kv = {"screen_name": username, "withSafetyModeUserFields": True}
    ft = {
        "highlights_tweets_tab_ui_enabled": True,
        "hidden_profile_likes_enabled": True,
        "creator_subscriptions_tweet_preview_api_enabled": True,
        "hidden_profile_subscriptions_enabled": True,
        "subscriptions_verification_info_verified_since_enabled": True,
        "subscriptions_verification_info_is_identity_verified_enabled": False,
        "responsive_web_twitter_article_notes_tab_enabled": False,
        "subscriptions_feature_can_gift_premium": False,
        "profile_label_improvements_pcf_label_in_post_enabled": False,
    }
    params = encode_params({"variables": kv, "features": ft})

    url = f"{GQL_URL}/{OP_USER_BY_SCREEN_NAME}"
    path = f"/i/api/graphql/{OP_USER_BY_SCREEN_NAME}"
    headers = {
        "x-client-transaction-id": clid_gen.calc("GET", path),
        "content-type": "application/json",
    }

    resp = await client.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    try:
        return data["data"]["user"]["result"]["rest_id"]
    except (KeyError, TypeError):
        return None


async def follow_user(client, clid_gen, user_id: str, username: str = "") -> dict:
    """Send a v1.1 friendships/create POST to follow a user."""
    form_data = {**FOLLOW_FORM_FIELDS, "user_id": user_id}

    headers = {
        "x-client-transaction-id": clid_gen.calc("POST", FOLLOW_PATH),
    }
    if username:
        headers["referer"] = f"https://x.com/{username}"

    resp = await client.post(FOLLOW_URL, data=form_data, headers=headers)

    # Handle known non-error 403s (e.g. already requested to follow protected account)
    if resp.status_code == 403:
        try:
            body = resp.json()
            errors = body.get("errors", [])
            for err in errors:
                # Code 160: "You've already requested to follow X."
                if err.get("code") == 160:
                    logger.info(f"  Already requested (pending): {err['message']}")
                    return body
        except Exception:
            pass
        logger.error(f"  Response body: {resp.text[:500]}")

    resp.raise_for_status()
    return resp.json()


# ── Main ────────────────────────────────────────────────────────────────────

async def main(
    dry_run: bool = False,
    delay_min: int = 30,
    delay_max: int = 60,
    limit: int | None = None,
    resolve_only: bool = False,
):
    start_time = time.time()

    print("=" * 60)
    print("FOLLOW USERS")
    print("=" * 60)

    # 1. Load target usernames
    targets = load_target_usernames(LEADERBOARD_FILE)
    print(f"Leaderboard accounts: {len(targets)}")

    # 2. Get authenticated client directly (bypass twscrape queue system entirely)
    from twscrape.accounts_pool import AccountsPool
    pool = AccountsPool(TWSCRAPE_DB)
    account = await pool.get(ACCOUNT_USERNAME)
    client = make_browser_client(account)
    clid_gen = await XClIdGenStore.get(account.username)

    # 3. Resolve AISafetyFeed user ID
    me_id = await resolve_user_id_raw(client, clid_gen, ACCOUNT_USERNAME)
    if not me_id:
        print("ERROR: Could not resolve AISafetyFeed user ID")
        return
    print(f"Account: @{ACCOUNT_USERNAME} (ID: {me_id})")

    # 4. Get current following list via twscrape (uses its own queue, separate from UserByScreenName)
    print("\nFetching current following list...")
    api = API(TWSCRAPE_DB)
    await api.pool.reset_locks()
    currently_following = set()
    id_cache = load_id_cache()
    count = 0
    async for u in api.following(int(me_id), limit=1000):
        uname = u.username.lower()
        currently_following.add(uname)
        id_cache[uname] = str(u.id)
        count += 1
        if count % 100 == 0:
            logger.info(f"  ...fetched {count} following")
    save_id_cache(id_cache)
    logger.info(f"Currently following {len(currently_following)} accounts")

    # 5. Compute diff
    to_follow = [u for u in targets if u not in currently_following]
    already = len(targets) - len(to_follow)
    print(f"\nAlready following: {already}")
    print(f"Need to follow:   {len(to_follow)}")

    if limit:
        to_follow = to_follow[:limit]
        print(f"Limited to:       {len(to_follow)}")

    if not to_follow:
        print("\nNothing to do!")
        return

    # 6. Resolve user IDs using raw client (bypasses twscrape queue/rate limits)
    print("\nResolving user IDs...")
    to_resolve = [u for u in to_follow if u not in id_cache]
    cached_count = len(to_follow) - len(to_resolve)
    if cached_count:
        logger.info(f"  {cached_count} already in cache")

    for i, username in enumerate(to_resolve, 1):
        try:
            uid = await resolve_user_id_raw(client, clid_gen, username)
            if uid:
                id_cache[username] = uid
                if i % 25 == 0:
                    logger.info(f"  ...resolved {i}/{len(to_resolve)}")
                    save_id_cache(id_cache)
            else:
                logger.warning(f"  @{username} - not found")
        except httpx.HTTPStatusError as e:
            logger.error(f"  @{username} - HTTP {e.response.status_code}: {e.response.text[:200]}")
            save_id_cache(id_cache)
            if e.response.status_code == 429:
                logger.error(f"  Rate limited after {i-1} resolves. Re-run to continue (cache saved).")
                break
        except Exception as e:
            logger.error(f"  @{username} - error: {e}")

    save_id_cache(id_cache)
    resolved_count = sum(1 for u in to_follow if u in id_cache)
    logger.info(f"Resolved {resolved_count}/{len(to_follow)} user IDs")

    if resolve_only:
        print(f"\n[RESOLVE ONLY] Cache saved with {len(id_cache)} entries.")
        return

    if dry_run:
        print(f"\n[DRY RUN] Would follow {len(to_follow)} accounts:")
        for i, username in enumerate(to_follow, 1):
            uid = id_cache.get(username, "???")
            print(f"  {i:3d}. @{username} (ID: {uid})")
        return

    # 7. Follow each user
    followable = [(u, id_cache[u]) for u in to_follow if u in id_cache]
    print(f"\nFollowing {len(followable)} accounts (delay: {delay_min}-{delay_max}s)...")
    print(f"Estimated time: {len(followable) * (delay_min + delay_max) / 2 / 60:.0f} minutes")
    print("-" * 60)

    succeeded = 0
    failed = 0

    for i, (username, user_id) in enumerate(followable, 1):
        try:
            resp = await follow_user(client, clid_gen, user_id, username)
            succeeded += 1
            logger.info(f"[{i}/{len(followable)}] @{username} (ID: {user_id}) - followed")

        except Exception as e:
            failed += 1
            logger.error(f"[{i}/{len(followable)}] @{username} - FAILED: {e}")

            # Refresh transaction ID on auth errors
            if "403" in str(e) or "401" in str(e):
                logger.info("  Refreshing transaction ID generator...")
                try:
                    clid_gen = await XClIdGenStore.get(account.username, fresh=True)
                except Exception as refresh_err:
                    logger.error(f"  Failed to refresh: {refresh_err}")

            # Back off on rate limit
            if "429" in str(e):
                wait = 900
                logger.info(f"  Rate limited! Waiting {wait}s...")
                await asyncio.sleep(wait)
                continue

        # Delay between follows (skip after last one)
        if i < len(followable):
            delay = random.uniform(delay_min, delay_max)
            logger.info(f"  Waiting {delay:.0f}s...")
            await asyncio.sleep(delay)

    # 8. Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Follow Twitter/X accounts from the leaderboard"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which accounts would be followed without actually following",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only resolve user IDs (build cache), don't follow",
    )
    parser.add_argument(
        "--delay-min",
        type=int,
        default=30,
        help="Minimum delay between follows in seconds (default: 30)",
    )
    parser.add_argument(
        "--delay-max",
        type=int,
        default=60,
        help="Maximum delay between follows in seconds (default: 60)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only follow the first N accounts",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            dry_run=args.dry_run,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
            limit=args.limit,
            resolve_only=args.resolve_only,
        )
    )
