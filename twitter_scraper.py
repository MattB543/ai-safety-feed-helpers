#!/usr/bin/env python3
"""
twitter_scraper.py
──────────────────
Scrape tweets from a list of Twitter profiles using twscrape.

This script:
1. Fetches the most recent tweets (including replies, retweets, quote tweets)
   from each profile in the hardcoded list
2. Hydrates parent tweets for replies (1 level up)
3. Saves everything to the PostgreSQL database

Usage:
    python twitter_scraper.py
    python twitter_scraper.py --limit 50  # Limit tweets per profile
    python twitter_scraper.py --debug     # Enable debug logging

Requirements:
    pip install twscrape psycopg2-binary python-dotenv
"""

import asyncio
import os
import sys
import logging
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass

import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv

# NOTE: twscrape >= 0.19.0 is required. X's web bundle changed several times in
# 2025-2026, repeatedly breaking the x-client-transaction-id generation. Older
# twscrape (<= 0.17.0) shipped a stale parser and this file used to carry an
# inline monkeypatch of `xclid.get_scripts_list` to work around it. Both are now
# obsolete: 0.18.0/0.19.0 rewrote the bundle parser and refreshed the GraphQL
# operation IDs. The old monkeypatch actively *re-broke* newer twscrape by
# forcing the outdated parser back, so it has been removed. If auth starts
# failing again with "Failed to parse scripts" / IndexError, first `pip install
# --upgrade twscrape`.
try:
    from twscrape import API
    from twscrape.logger import set_log_level
except ImportError:
    print("ERROR: twscrape not installed. Run: pip install 'twscrape>=0.19.1'")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

load_dotenv(override=True)

DATABASE_URL = os.getenv("AI_SAFETY_FEED_DB_URL")
TWSCRAPE_DB = "accounts.db"

if not DATABASE_URL:
    print("ERROR: AI_SAFETY_FEED_DB_URL environment variable not set.")
    sys.exit(1)

# ───── Hardcoded profile list ─────────────────────────────────────────────────
TARGET_PROFILES = [
    "jackclarksf",
    "ch402",
    "miles_brundage",
    "gwern",
    "michael_nielsen",
    "tdietterich",
    "janleike",
    "slatestarcodex",
    "amandaaskell",
    "oriolvinyalsml",
    "nandodf",
    "johnschulman2",
    "esyudkowsky",
    "sleepinyourhat",
    "chelseabfinn",
    "tszzl",
    "chrmanning",
    "kchonyc",
    "pabbeel",
    "hugo_larochelle",
    "garymarcus",
    "sarahookr",
    "soumithchintala",
    "jeremyphoward",
    "dwarkesh_sp",
    "chrszegedy",
    "polynoamial",
    "catherineols",
    "rsalakhu",
    "zacharylipton",
    "percyliang",
    "shakir_za",
    "davidduvenaud",
    "ericjang11",
    "tylercowen",
    "emollick",
    "richardmcngo",
    "shanelegg",
    "natfriedman",
    # "danhendrycks",  # DROPPED — renamed to "hendrycks", which is already in this list below (scraped fine)
    "lilianweng",
    "thom_wolf",
    "kelseytuoc",
    "srush_nlp",
    "erikbryn",
    "svlevine",
    "emostaque",
    "goodside",
    "richardsocher",
    "willmacaskill",
    "jacobsteinhardt",
    "random_walker",
    "ajeya_cotra",
    "hannawallach",
    "erichorvitz",
    "quocleix",
    "dpkingma",
    "robinhanson",
    "melmitchell1",
    "hlntnr",
    "sebastienbubeck",
    "_jasonwei",
    "jekbradbury",
    "mustafasuleyman",
    "davidskrueger",
    "katjagrace",
    "jachiam0",
    "liamfedus",
    "vkrakovna",
    "nearcyan",
    "robertwiblin",
    "eshear",
    "albrgr",
    "markchen90",
    "davidad",
    "jkcarlsmith",
    "jasoncrawford",
    "yoshua_bengio",
    "sedielem",
    "lukeprog",
    "jacobandreas",
    "neelnanda5",
    "arankomatsuzaki",
    "clementdelangue",
    "zoubinghahrama1",
    "jeffclune",
    "roydanroy",
    "barret_zoph",
    "davidsholz",
    "mmitchell_ai",
    "aleks_madry",
    "_rockt",
    "owainevans_uk",
    "shaneguml",
    "yejinchoinka",
    "zicokolter",
    "poolio",
    "_beenkim",
    "suchenzang",
    "drmichaellevin",
    "egrefen",
    "nickcammarata",
    "smerity",
    "geoffreyirving",
    "ibab",
    "_sholtodouglas",
    "beenwrekt",
    "dustinvtran",
    "natolambert",
    "ethanjperez",
    "typedfemale",
    "yoavgo",
    "jack_w_rae",
    "patio11",
    "eladgil",
    "tegmark",
    "natashajaques",
    "sebkrier",
    "tim_dettmers",
    "rivatez",
    "danilojrezende",
    "dkokotajlo",
    "anderssandberg",
    "giffmana",
    "boazbaraktcs",
    "tobyordoxford",
    "rajiinio",
    "stanislavfort",
    "richardssutton",
    "mer__edith",
    "shivon",
    "csvoss",
    "doomie",
    "singularmattrix",
    "alecstapp",
    "bobmcgrewai",
    "nabla_theta",
    "evanhub",
    "maithra_raghu",
    "tri_dao",
    "nalkalc",
    "wellingmax",
    "iamtrask",
    "prfsanjeevarora",
    "savvyrl",
    "kevinroose",
    "seb_ruder",
    "thezvi",
    "simonw",
    "laurademing",
    "ashvaswani",
    "timnitgebru",
    "kipperrii",
    "pfau",
    "blancheminerva",
    "dylan522p",
    "deliprao",
    "tengyuma",
    "webdevmason",
    "artirkel",
    # "etzioni",  # DROPPED — Oren Etzioni deleted his X account (per his LinkedIn: "I cancelled my Twitter/X account")
    "npcollapse",
    "repligate",
    "officiallogank",
    "tacocohen",
    "_karenhao",
    "yaringal",
    "chhillee",
    "jeffladish",
    "spencrgreenberg",
    "randomlywalking",
    "aidangomez",
    "deanwball",
    "ptetlock",
    "j_foerst",
    "soundboy",
    "timhwang",
    "mark_riedl",
    "thegregyang",
    "stefanfschubert",
    "liv_boeree",
    "rohinmshah",
    "adamdangelo",
    "chipro",
    "achowdhery",
    "avdnoord",
    "millionint",
    "plinz",
    "tracewoodgrains",
    "tamaybes",
    "matthewclifford",
    "rgblong",
    "yudapearl",
    "guillaumelample",
    "jaseweston",
    "gstsdn",
    "adammarblestone",
    "_arohan_",
    "karinanguyen",  # was karinanguyen_ (dropped trailing underscore; verified Karina Nguyen, ex-OpenAI/Anthropic)
    "maxjaderberg",
    "arthurmensch",
    "ben_j_todd",
    "__nmca__",
    "azaliamirh",
    "ancadianadragan",
    "rosiecampbell",
    "dhadfieldmenell",
    "ohabryka",
    "sirbayes",
    "collision",
    "joannejang",
    "rapha_gl",
    "argleave",
    "so8res",
    "alexeyguzey",
    "jam3scampbell",
    "atabarrok",
    "jefrankle",
    "peterwildeford",
    "robbensinger",
    "mpshanahan",
    "npew",
    "_aidan_clark_",
    "bshlgrs",
    "gneubig",
    "yonashav",
    "tejasdkulkarni",
    "bneyshabur",
    "willdepue",
    "tomgoldsteincs",
    "prafdhar",
    "jesswhittles",
    "rodneyabrooks",
    "Teknium",  # was teknium1 (dropped the "1"; verified Nous Research co-founder)
    "matthewjbar",
    "saranormous",
    "ykilcher",
    "cjmaddison",
    "robertskmiles",
    "kanjun",
    "davidmanheim",
    "elder_plinius",
    "yitayml",
    "model_mechanic",
    "salonium",
    "meaningness",
    "lmthang",
    "pushmeet",
    "adversariel",
    "stuartjritchie",
    "samoburja",
    "andrewgwils",
    "koraykv",
    "ryan_t_lowe",
    "teortaxestex",
    "swyx",
    "norabelrose",
    "davidchalmers42",
    "aidan_mclau",
    "borismpower",
    "hamandcheese",
    "khoomeik",
    "atroyn",
    "denny_zhou",
    "nostalgebraist",
    "haydnbelfield",
    "rishibommasani",
    "ohlennart",
    "mealreplacer",
    "mparakhin",
    "ashleevance",
    "labenz",
    "jxmnop",
    "mobav0",
    "s8mb",
    "dawnsongtweets",
    "bethmaybarnes",
    "manderljung",
    "merettm",
    "iasongabriel",
    "alextamkin",
    "yuhu_ai_",
    # "jacobmenick",  # DROPPED — account deactivated/deleted (handle and own tweets no longer resolve)
    "logangraham",
    "tshevl",
    "eli_lifland",
    "andrewlampinen",
    "s_oheigeartaigh",
    "dileeplearning",
    "trentonbricken",
    "ethancaballero",
    "gmiller",  # was primalpoly (renamed; verified Geoffrey Miller, 152k followers)
    "sherjilozair",
    "danijarh",
    "irenesolaiman",
    "lxrjl",
    "aisafetymemes",
    "kenneth0stanley",
    "oh_that_hat",
    "nathanpmyoung",
    "altimor",
    "andrewcritchphd",
    "andrewcurran_",
    "anthropicai",
    "apolloaievals",
    "arthurb",
    "benlandautaylor",
    "binarybits",
    "cais",
    "charlesmonneron",
    "connoraxiotes",
    "danfaggella",
    "daniel_271828",
    "deedydas",
    "dfrsrchtwts",
    "dorialexander",
    "epistemichope",
    "epochairesearch",
    "gallabytes",
    "garrisonlovely",
    "gdb",
    "googledeepmind",
    "hendrycks",
    "jd_pressman",
    "jjding99",
    "karpathy",
    "KatSpartz",  # was kat__woods (renamed to Kat Spartz; verified AI-safety bio, Nonlinear co-founder)
    "krishnanrohit",
    "leopoldasch",
    "liron",
    "mariushobbhahn",
    "michaeltrazzi",
    "nabeelqu",
    "natesilver538",
    "nunosempere",
    "ollie_base",
    "openai",
    "oscredwin",
    "ozziegooen",
    "quintinpope5",
    "r_zwetsloot",
    "rao2z",
    "s_r_constantin",
    "sethburn",
    "shakeelhashim",
    "simeon_cps",
    "thomas_woodside",
    "tsarnick",
    "turn_trout",
    "tyler_m_john",
]

# ───── Scraper settings ───────────────────────────────────────────────────────
DEFAULT_TWEETS_PER_PROFILE = 100

# When a --since cutoff is given, stop paging a profile after this many
# consecutive tweets older than the cutoff. A tolerance (rather than stopping on
# the first old tweet) is required because user_tweets_and_replies is only
# roughly reverse-chronological: pinned tweets surface at the top and hydrated
# reply context can interleave out of order.
SINCE_STOP_STREAK = 30

# ───── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScraperStats:
    """Track scraping statistics."""
    profiles_attempted: int = 0
    profiles_succeeded: int = 0
    tweets_scraped: int = 0
    tweets_inserted: int = 0
    tweets_skipped_duplicate: int = 0
    parents_hydrated: int = 0
    errors: int = 0


def classify_tweet(tweet) -> str:
    """Determine the interaction type of a tweet."""
    if tweet.retweetedTweet:
        return "retweet"
    elif tweet.quotedTweet:
        return "quote_tweet"
    elif tweet.inReplyToTweetId:
        return "reply"
    else:
        return "tweet"


def tweet_to_db_tuple(
    tweet,
    source_profile: Optional[str] = None,
    is_hydrated_parent: bool = False
) -> tuple:
    """
    Convert a twscrape Tweet object to a database tuple.

    Returns a tuple matching the INSERT statement column order.
    """
    interaction_type = classify_tweet(tweet)

    # Extract URLs from links (external URLs in the tweet)
    urls = []
    if tweet.links:
        urls.extend([link.url for link in tweet.links if link and link.url])

    # Also extract URLs from retweeted content
    if tweet.retweetedTweet and tweet.retweetedTweet.links:
        urls.extend([link.url for link in tweet.retweetedTweet.links if link and link.url])

    # Also extract URLs from quoted content
    if tweet.quotedTweet and tweet.quotedTweet.links:
        urls.extend([link.url for link in tweet.quotedTweet.links if link and link.url])

    # Deduplicate URLs while preserving order
    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]

    # Extract media URLs (photos, video thumbnails, GIFs)
    media_urls = []
    if tweet.media:
        if tweet.media.photos:
            media_urls.extend([p.url for p in tweet.media.photos if p and p.url])
        if tweet.media.videos:
            media_urls.extend([v.thumbnailUrl for v in tweet.media.videos if v and v.thumbnailUrl])
            # Also store actual video URLs (best quality)
            for v in tweet.media.videos:
                if v.variants:
                    # Get highest bitrate video
                    best = max(v.variants, key=lambda x: x.bitrate or 0)
                    if best.url:
                        media_urls.append(best.url)
        if tweet.media.animated:  # GIFs
            for anim in tweet.media.animated:
                if hasattr(anim, 'videoUrl') and anim.videoUrl:
                    media_urls.append(anim.videoUrl)
                elif hasattr(anim, 'url') and anim.url:
                    media_urls.append(anim.url)

    # Also extract media from retweeted content
    if tweet.retweetedTweet and tweet.retweetedTweet.media:
        rt_media = tweet.retweetedTweet.media
        if rt_media.photos:
            media_urls.extend([p.url for p in rt_media.photos if p and p.url])
        if rt_media.videos:
            media_urls.extend([v.thumbnailUrl for v in rt_media.videos if v and v.thumbnailUrl])
            for v in rt_media.videos:
                if v.variants:
                    best = max(v.variants, key=lambda x: x.bitrate or 0)
                    if best.url:
                        media_urls.append(best.url)

    # Also extract media from quoted content
    if tweet.quotedTweet and tweet.quotedTweet.media:
        qt_media = tweet.quotedTweet.media
        if qt_media.photos:
            media_urls.extend([p.url for p in qt_media.photos if p and p.url])
        if qt_media.videos:
            media_urls.extend([v.thumbnailUrl for v in qt_media.videos if v and v.thumbnailUrl])

    # Deduplicate media URLs
    seen_media = set()
    media_urls = [u for u in media_urls if not (u in seen_media or seen_media.add(u))]

    # Build the tuple (order must match INSERT_SQL columns)
    return (
        str(tweet.id),                                          # tweet_id
        str(tweet.conversationId) if tweet.conversationId else None,  # conversation_id

        # Author info
        tweet.user.username if tweet.user else None,            # author_username
        str(tweet.user.id) if tweet.user else None,             # author_id
        tweet.user.displayname if tweet.user else None,         # author_display_name

        # Content
        tweet.rawContent or "",                                 # text
        tweet.date,                                             # created_at

        # Classification
        interaction_type,                                       # interaction_type

        # Reply relationships
        str(tweet.inReplyToTweetId) if tweet.inReplyToTweetId else None,  # in_reply_to_tweet_id
        tweet.inReplyToUser.username if tweet.inReplyToUser else None,    # in_reply_to_username
        str(tweet.inReplyToUser.id) if tweet.inReplyToUser else None,     # in_reply_to_user_id

        # Quote tweet relationships
        str(tweet.quotedTweet.id) if tweet.quotedTweet else None,                           # quoted_tweet_id
        tweet.quotedTweet.rawContent if tweet.quotedTweet else None,                        # quoted_text
        tweet.quotedTweet.user.username if tweet.quotedTweet and tweet.quotedTweet.user else None,  # quoted_username
        str(tweet.quotedTweet.user.id) if tweet.quotedTweet and tweet.quotedTweet.user else None,   # quoted_user_id

        # Retweet relationships
        str(tweet.retweetedTweet.id) if tweet.retweetedTweet else None,                              # retweeted_tweet_id
        tweet.retweetedTweet.rawContent if tweet.retweetedTweet else None,                           # retweeted_text
        tweet.retweetedTweet.user.username if tweet.retweetedTweet and tweet.retweetedTweet.user else None,  # retweeted_username
        str(tweet.retweetedTweet.user.id) if tweet.retweetedTweet and tweet.retweetedTweet.user else None,   # retweeted_user_id

        # Engagement metrics
        tweet.likeCount or 0,                                   # favorite_count
        tweet.retweetCount or 0,                                # retweet_count
        tweet.replyCount or 0,                                  # reply_count
        tweet.quoteCount or 0,                                  # quote_count
        tweet.viewCount,                                        # view_count (can be None)

        # URLs and media (PostgreSQL arrays)
        urls if urls else None,                                 # urls
        media_urls if media_urls else None,                     # media_urls

        # Scraping metadata
        source_profile,                                         # source_profile
        is_hydrated_parent,                                     # is_hydrated_parent
        datetime.now(timezone.utc),                             # scraped_at
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

INSERT_TWEET_SQL = """
INSERT INTO tweets (
    tweet_id, conversation_id,
    author_username, author_id, author_display_name,
    text, created_at,
    interaction_type,
    in_reply_to_tweet_id, in_reply_to_username, in_reply_to_user_id,
    quoted_tweet_id, quoted_text, quoted_username, quoted_user_id,
    retweeted_tweet_id, retweeted_text, retweeted_username, retweeted_user_id,
    favorite_count, retweet_count, reply_count, quote_count, view_count,
    urls, media_urls,
    source_profile, is_hydrated_parent, scraped_at
) VALUES (
    %s, %s,
    %s, %s, %s,
    %s, %s,
    %s,
    %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s,
    %s, %s, %s
)
ON CONFLICT (tweet_id) DO NOTHING;
"""

UPSERT_PROFILE_SQL = """
INSERT INTO twitter_profiles (username, user_id, display_name, follower_count, following_count, description, last_scraped_at)
VALUES (%s, %s, %s, %s, %s, %s, NOW())
ON CONFLICT (username) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    display_name = EXCLUDED.display_name,
    follower_count = EXCLUDED.follower_count,
    following_count = EXCLUDED.following_count,
    description = EXCLUDED.description,
    last_scraped_at = NOW(),
    updated_at = NOW();
"""


def connect_db():
    """
    Open a DB connection with TCP keepalives.

    Keepalives keep the single long-lived connection from being dropped by the
    managed-Postgres/pooler idle timeout during the multi-minute rate-limit
    waits in the hydration phase (seen as "connection already closed" on long
    --limit runs). They are necessary but NOT sufficient — DO's load balancer /
    NAT still drops idle conns — so callers must also reconnect on failure via
    ensure_conn() / insert_tweets_resilient().
    """
    return psycopg2.connect(
        DATABASE_URL,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )


def ensure_conn(conn):
    """
    Return a live connection, transparently reconnecting if the current one was
    dropped during a long rate-limit wait.

    Pings with SELECT 1 before a write batch; if the handle is closed or the
    ping fails, opens a fresh connection. The in-memory existing-tweet-id set is
    held by the caller and untouched here, so it survives a reconnect.
    """
    try:
        if conn is not None and conn.closed == 0:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return conn
    except psycopg2.Error:
        pass  # fall through to reconnect

    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    logger.warning("   DB connection lost — reconnecting")
    return connect_db()


def insert_tweets(conn, tweet_tuples: List[tuple]) -> tuple[int, int]:
    """
    Insert tweets into database.

    Returns: (inserted_count, skipped_count)
    """
    if not tweet_tuples:
        return 0, 0

    inserted = 0
    skipped = 0

    with conn.cursor() as cur:
        for tweet_tuple in tweet_tuples:
            try:
                cur.execute(INSERT_TWEET_SQL, tweet_tuple)
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1  # ON CONFLICT DO NOTHING
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                # Connection-level failure ("connection already closed",
                # "cursor already closed"): abort the batch so the caller can
                # reconnect and retry it wholesale, instead of logging one
                # warning per remaining tweet.
                raise
            except psycopg2.Error as e:
                logger.warning(f"Failed to insert tweet: {e}")
                skipped += 1

    conn.commit()
    return inserted, skipped


def insert_tweets_resilient(conn, tweet_tuples: List[tuple]) -> tuple:
    """
    insert_tweets with a single reconnect-and-retry on a dropped connection.

    Returns (conn, inserted, skipped); `conn` may be a fresh handle. Safe to
    retry the whole batch because the INSERT is ON CONFLICT DO NOTHING.
    """
    conn = ensure_conn(conn)
    try:
        inserted, skipped = insert_tweets(conn, tweet_tuples)
        return conn, inserted, skipped
    except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
        logger.warning(f"   Insert failed ({e}); reconnecting and retrying batch once")
        conn = connect_db()
        inserted, skipped = insert_tweets(conn, tweet_tuples)
        return conn, inserted, skipped


def upsert_profile(conn, user) -> None:
    """Update or insert profile information."""
    if not user:
        return

    with conn.cursor() as cur:
        cur.execute(UPSERT_PROFILE_SQL, (
            user.username,
            str(user.id),
            user.displayname,
            user.followersCount,
            user.friendsCount,
            user.rawDescription,
        ))
    conn.commit()


def get_existing_tweet_ids(conn) -> Set[str]:
    """Get all existing tweet IDs from database."""
    with conn.cursor() as cur:
        cur.execute("SELECT tweet_id FROM tweets")
        return {row[0] for row in cur.fetchall()}


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

async def scrape_profile(
    api: API,
    username: str,
    limit: int,
    existing_ids: Set[str],
    since: Optional[datetime] = None,
) -> tuple[List[tuple], Set[str], Any]:
    """
    Scrape tweets from a single profile.

    Returns:
        - List of tweet tuples ready for DB insertion
        - Set of parent tweet IDs that need hydration
        - User object (or None)
    """
    logger.info(f"{'-' * 50}")
    logger.info(f"Scraping @{username}")

    tweet_tuples = []
    parent_ids_to_hydrate: Set[str] = set()
    user = None

    try:
        # Resolve username to user object
        user = await api.user_by_login(username)
        if not user:
            logger.error(f"   Could not find user @{username}")
            return [], set(), None

        logger.info(f"   Found @{username} (ID: {user.id}, Followers: {user.followersCount:,})")

        # Scrape tweets
        tweet_count = 0
        old_streak = 0
        stats = {"tweet": 0, "reply": 0, "quote_tweet": 0, "retweet": 0}

        async for tweet in api.user_tweets_and_replies(user.id, limit=limit):
            tweet_id = str(tweet.id)

            # Date cutoff for "fresh only" scrapes: skip tweets older than `since`
            # and give up on the profile once we hit a run of old tweets.
            if since is not None and tweet.date is not None and tweet.date < since:
                old_streak += 1
                if old_streak >= SINCE_STOP_STREAK:
                    logger.info(
                        f"   Hit {SINCE_STOP_STREAK} consecutive tweets older than "
                        f"{since.date()}, stopping early for @{username}"
                    )
                    break
                continue
            old_streak = 0

            # Skip if already in database
            if tweet_id in existing_ids:
                continue

            # Convert to DB tuple
            tweet_tuple = tweet_to_db_tuple(tweet, source_profile=username)
            tweet_tuples.append(tweet_tuple)

            # Track interaction type
            interaction_type = classify_tweet(tweet)
            stats[interaction_type] += 1

            # Collect parent IDs for hydration (replies only - quotes/RTs have inline content)
            if tweet.inReplyToTweetId:
                parent_id = str(tweet.inReplyToTweetId)
                if parent_id not in existing_ids:
                    parent_ids_to_hydrate.add(parent_id)

            tweet_count += 1

            # Progress indicator
            if tweet_count % 25 == 0:
                logger.info(f"   ... {tweet_count} tweets collected")

        logger.info(f"   Collected {len(tweet_tuples)} new tweets")
        logger.info(f"      Breakdown: {stats['tweet']} original, {stats['reply']} replies, "
                    f"{stats['quote_tweet']} quotes, {stats['retweet']} RTs")
        logger.info(f"      Parent tweets to hydrate: {len(parent_ids_to_hydrate)}")

    except Exception as e:
        logger.error(f"   Error scraping @{username}: {e}")
        import traceback
        traceback.print_exc()

    return tweet_tuples, parent_ids_to_hydrate, user


async def hydrate_parent_tweets(
    api: API,
    parent_ids: Set[str],
    existing_ids: Set[str]
) -> List[tuple]:
    """
    Fetch parent tweets for hydration.

    Returns list of tweet tuples ready for DB insertion.
    """
    # Filter out IDs we already have
    ids_to_fetch = parent_ids - existing_ids

    if not ids_to_fetch:
        return []

    logger.info(f"\n{'-' * 50}")
    logger.info(f"Hydrating {len(ids_to_fetch)} parent tweets")

    tweet_tuples = []
    fetched = 0
    failed = 0

    for tweet_id in ids_to_fetch:
        try:
            tweet = await api.tweet_details(int(tweet_id))

            if tweet:
                tweet_tuple = tweet_to_db_tuple(
                    tweet,
                    source_profile=None,  # Not from our profile list
                    is_hydrated_parent=True
                )
                tweet_tuples.append(tweet_tuple)
                fetched += 1

                if fetched % 10 == 0:
                    logger.info(f"   ... {fetched}/{len(ids_to_fetch)} hydrated")
            else:
                failed += 1

        except Exception as e:
            logger.warning(f"   Failed to hydrate tweet {tweet_id}: {e}")
            failed += 1

    logger.info(f"   Hydrated {fetched} tweets ({failed} failed/unavailable)")
    return tweet_tuples


async def run_scraper(
    limit: int = DEFAULT_TWEETS_PER_PROFILE,
    start_from: int = 1,
    since: Optional[datetime] = None,
    profiles: Optional[List[str]] = None,
) -> ScraperStats:
    """Main scraper execution."""
    stats = ScraperStats()
    start_time = time.time()

    # `profiles` (from --profiles) overrides the hardcoded list for a targeted
    # run; otherwise scrape the full TARGET_PROFILES list.
    source_list = profiles if profiles else TARGET_PROFILES

    # Validate start_from
    start_from = max(1, min(start_from, len(source_list)))
    profiles_to_scrape = source_list[start_from - 1:]

    print("\n" + "=" * 60)
    print("TWITTER PROFILE SCRAPER")
    print("=" * 60)
    print(f"Profiles to scrape: {len(profiles_to_scrape)} (starting from #{start_from})")
    print(f"Tweets per profile: {limit}")
    print(f"Since cutoff: {since.date().isoformat() if since else '(none — full limit)'}")
    print(f"twscrape DB: {TWSCRAPE_DB}")
    print("=" * 60)

    # Initialize twscrape API
    api = API(TWSCRAPE_DB)

    # Check account pool
    try:
        pool_stats = await api.pool.stats()
        logger.info(f"\nAccount Pool: {pool_stats}")

        if pool_stats.get('active', 0) == 0:
            logger.error("No active Twitter accounts!")
            logger.error("   Run: python setup_twitter_accounts.py")
            return stats
    except Exception as e:
        logger.error(f"Failed to check account pool: {e}")
        return stats

    # Connect to database
    conn = None
    try:
        logger.info("\nConnecting to database...")
        conn = connect_db()
        logger.info("   Connected")

        # Get existing tweet IDs to avoid duplicates
        existing_ids = get_existing_tweet_ids(conn)
        logger.info(f"   Found {len(existing_ids):,} existing tweets in DB")

        # Scrape each profile and save immediately
        for i, username in enumerate(profiles_to_scrape, start_from):
            stats.profiles_attempted += 1

            logger.info(f"\n[{i}/{len(source_list)}] Processing @{username}")

            tweet_tuples, parent_ids, user = await scrape_profile(
                api, username, limit, existing_ids, since=since
            )

            if tweet_tuples:
                # Save tweets IMMEDIATELY (crash-safe). insert_tweets_resilient
                # pings + reconnects if the long scrape/hydration waits dropped
                # the connection, and may hand back a fresh `conn`.
                conn, inserted, skipped = insert_tweets_resilient(conn, tweet_tuples)
                stats.tweets_inserted += inserted
                stats.tweets_skipped_duplicate += skipped

                # Update existing_ids with newly inserted tweets
                for t in tweet_tuples:
                    existing_ids.add(t[0])  # t[0] is tweet_id

                # Hydrate and save parent tweets for THIS profile
                parents_inserted = 0
                if parent_ids:
                    parent_tuples = await hydrate_parent_tweets(api, parent_ids, existing_ids)
                    if parent_tuples:
                        conn, p_inserted, p_skipped = insert_tweets_resilient(conn, parent_tuples)
                        stats.tweets_inserted += p_inserted
                        stats.parents_hydrated += p_inserted
                        parents_inserted = p_inserted
                        # Update existing_ids with hydrated parents
                        for t in parent_tuples:
                            existing_ids.add(t[0])

                stats.profiles_succeeded += 1

                # Update profile info in DB (non-critical metadata — never let
                # it abort the run if the connection just dropped).
                if user:
                    conn = ensure_conn(conn)
                    try:
                        upsert_profile(conn, user)
                    except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                        logger.warning(f"   Profile metadata upsert failed: {e}")
                        conn = connect_db()

                logger.info(f"   Saved {inserted} tweets, {parents_inserted} parents to DB")

            stats.tweets_scraped += len(tweet_tuples)

    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        stats.errors += 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        stats.errors += 1
    finally:
        if conn:
            conn.close()
            logger.info("\nDatabase connection closed")

    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"   Profiles: {stats.profiles_succeeded}/{stats.profiles_attempted} successful")
    print(f"   Tweets scraped: {stats.tweets_scraped}")
    print(f"   Parents hydrated: {stats.parents_hydrated}")
    print(f"   Tweets inserted: {stats.tweets_inserted}")
    print(f"   Duplicates skipped: {stats.tweets_skipped_duplicate}")
    print(f"   Errors: {stats.errors}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    print("=" * 60)

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape tweets from target Twitter profiles"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=DEFAULT_TWEETS_PER_PROFILE,
        help=f"Max tweets per profile (default: {DEFAULT_TWEETS_PER_PROFILE})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--start-from", "-s",
        type=int,
        default=1,
        help="Start from profile number N (1-indexed, default: 1)"
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only keep tweets on/after this UTC date (YYYY-MM-DD). Older tweets "
             "are skipped, and paging a profile stops after a run of old tweets. "
             "Use for a 'fresh only' scrape, e.g. --since 2026-03-05."
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=None,
        help="Comma-separated handles to scrape instead of the full TARGET_PROFILES "
             "list (e.g. --profiles hendrycks,teknium). Reuses the same per-profile "
             "pipeline; handy for targeted re-scrapes of recovered/renamed accounts."
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        set_log_level("DEBUG")

    since_dt = None
    if args.since:
        try:
            since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: --since must be YYYY-MM-DD, got: {args.since!r}")
            sys.exit(1)

    profiles = None
    if args.profiles:
        profiles = [h.strip().lstrip("@") for h in args.profiles.split(",") if h.strip()]

    # Run the async scraper
    asyncio.run(run_scraper(
        limit=args.limit, start_from=args.start_from, since=since_dt, profiles=profiles
    ))


if __name__ == "__main__":
    main()
