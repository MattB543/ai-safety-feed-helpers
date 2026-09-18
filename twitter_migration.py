#!/usr/bin/env python3
"""
twitter_migration.py
────────────────────
Database migration script to create tables for Twitter scraping.

Creates:
  - twitter_profiles: Tracks profiles we're scraping
  - tweets: Stores all scraped tweets and hydrated parent tweets

Run this script once before using the Twitter scraper.

Usage:
    python twitter_migration.py
"""

import os
import sys
import logging
import psycopg2
from dotenv import load_dotenv

# ───── Load environment variables ─────────────────────────────────────────────
load_dotenv(override=True)

DATABASE_URL = os.getenv("AI_SAFETY_FEED_DB_URL")

if not DATABASE_URL:
    print("ERROR: AI_SAFETY_FEED_DB_URL environment variable not set.")
    sys.exit(1)

# ───── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)

# ───── Migration SQL ──────────────────────────────────────────────────────────

CREATE_TWITTER_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS twitter_profiles (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    user_id TEXT,
    display_name TEXT,
    follower_count INTEGER,
    following_count INTEGER,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    last_scraped_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_TWEETS_TABLE = """
CREATE TABLE IF NOT EXISTS tweets (
    id SERIAL PRIMARY KEY,
    tweet_id TEXT UNIQUE NOT NULL,
    conversation_id TEXT,

    -- Author info
    author_username TEXT NOT NULL,
    author_id TEXT,
    author_display_name TEXT,

    -- Content
    text TEXT,
    created_at TIMESTAMPTZ,

    -- Classification
    interaction_type TEXT CHECK (interaction_type IN ('tweet', 'reply', 'retweet', 'quote_tweet')),

    -- Relationships (for replies)
    in_reply_to_tweet_id TEXT,
    in_reply_to_username TEXT,
    in_reply_to_user_id TEXT,

    -- Relationships (for quote tweets)
    quoted_tweet_id TEXT,
    quoted_text TEXT,
    quoted_username TEXT,
    quoted_user_id TEXT,

    -- Relationships (for retweets)
    retweeted_tweet_id TEXT,
    retweeted_text TEXT,
    retweeted_username TEXT,
    retweeted_user_id TEXT,

    -- Engagement metrics
    favorite_count INTEGER DEFAULT 0,
    retweet_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    quote_count INTEGER DEFAULT 0,
    view_count INTEGER,

    -- URLs and media (stored as arrays)
    urls TEXT[],
    media_urls TEXT[],

    -- Scraping metadata
    source_profile TEXT,
    is_hydrated_parent BOOLEAN DEFAULT FALSE,
    scraped_at TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_CONVERSATION_CLASSIFICATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS twitter_conversation_classifications (
    conversation_id TEXT PRIMARY KEY,
    is_ai_safety BOOLEAN NOT NULL,
    classified_at TIMESTAMPTZ DEFAULT NOW(),
    model_used TEXT DEFAULT 'gemini-2.5-flash'
);
"""

CREATE_INDEXES = """
-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_tweets_tweet_id ON tweets(tweet_id);
CREATE INDEX IF NOT EXISTS idx_tweets_author_username ON tweets(author_username);
CREATE INDEX IF NOT EXISTS idx_tweets_source_profile ON tweets(source_profile);
CREATE INDEX IF NOT EXISTS idx_tweets_created_at ON tweets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tweets_interaction_type ON tweets(interaction_type);
CREATE INDEX IF NOT EXISTS idx_tweets_conversation_id ON tweets(conversation_id);
CREATE INDEX IF NOT EXISTS idx_tweets_in_reply_to ON tweets(in_reply_to_tweet_id);
CREATE INDEX IF NOT EXISTS idx_twitter_profiles_username ON twitter_profiles(username);
CREATE INDEX IF NOT EXISTS idx_conv_class_is_ai_safety ON twitter_conversation_classifications(is_ai_safety);
"""

# ───── Migration execution ────────────────────────────────────────────────────

def run_migration():
    """Execute the database migration."""
    conn = None
    try:
        logging.info("Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False

        with conn.cursor() as cur:
            # Create twitter_profiles table
            logging.info("Creating twitter_profiles table...")
            cur.execute(CREATE_TWITTER_PROFILES_TABLE)

            # Create tweets table
            logging.info("Creating tweets table...")
            cur.execute(CREATE_TWEETS_TABLE)

            # Create conversation classifications table
            logging.info("Creating twitter_conversation_classifications table...")
            cur.execute(CREATE_CONVERSATION_CLASSIFICATIONS_TABLE)

            # Create indexes
            logging.info("Creating indexes...")
            cur.execute(CREATE_INDEXES)

            # Commit all changes
            conn.commit()
            logging.info("Migration completed successfully!")

            # Verify tables exist
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('twitter_profiles', 'tweets', 'twitter_conversation_classifications')
                ORDER BY table_name;
            """)
            tables = cur.fetchall()
            logging.info(f"Verified tables exist: {[t[0] for t in tables]}")

            # Show row counts
            for table in ['twitter_profiles', 'tweets', 'twitter_conversation_classifications']:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                logging.info(f"  {table}: {count} rows")

    except psycopg2.Error as e:
        logging.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        if conn:
            conn.rollback()
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


def drop_tables():
    """Drop the Twitter tables (for development/reset purposes)."""
    conn = None
    try:
        logging.info("Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False

        with conn.cursor() as cur:
            logging.warning("Dropping twitter_conversation_classifications table...")
            cur.execute("DROP TABLE IF EXISTS twitter_conversation_classifications CASCADE;")

            logging.warning("Dropping tweets table...")
            cur.execute("DROP TABLE IF EXISTS tweets CASCADE;")

            logging.warning("Dropping twitter_profiles table...")
            cur.execute("DROP TABLE IF EXISTS twitter_profiles CASCADE;")

            conn.commit()
            logging.info("Tables dropped successfully.")

    except psycopg2.Error as e:
        logging.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Database migration for Twitter scraping tables"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing tables before creating (WARNING: destroys data)"
    )
    parser.add_argument(
        "--drop-only",
        action="store_true",
        help="Only drop tables, don't recreate (WARNING: destroys data)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Twitter Database Migration")
    print("=" * 60)

    if args.drop_only:
        confirm = input("This will DELETE all Twitter data. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            drop_tables()
        else:
            print("Aborted.")
    elif args.drop:
        confirm = input("This will DELETE all Twitter data and recreate tables. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            drop_tables()
            run_migration()
        else:
            print("Aborted.")
    else:
        run_migration()
