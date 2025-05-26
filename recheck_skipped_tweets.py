#!/usr/bin/env python3
"""
Standalone Re-checker for Skipped Tweets
========================================

This script can be run periodically (e.g., via cron) to re-check previously
skipped tweets that may have gained popularity since they were originally filtered.

Usage:
    python recheck_skipped_tweets.py [--threshold 50] [--dry-run]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the current directory to the path so we can import from twitter_scraper
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from twitter.scraper import Scraper
from twitter_scraper import (
    recheck_skipped_tweets, 
    get_skipped_tweets_stats,
    OpenAI,
    openai_client
)


def main():
    # Load environment variables
    load_dotenv()
    
    # Check database URL
    DB_URL = os.getenv("AI_SAFETY_TWEETS_DB_URL")
    if not DB_URL:
        print("[FATAL] AI_SAFETY_TWEETS_DB_URL not set - abort")
        sys.exit(1)
    
    # Initialize OpenAI client
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    global openai_client
    
    if OPENAI_API_KEY:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("[INFO] OpenAI client initialized for AI safety filtering")
        except Exception as e:
            print(f"[WARNING] Failed to initialize OpenAI client: {e}")
            openai_client = None
    else:
        print("[WARNING] OPENAI_API_KEY not set - AI safety re-filtering disabled")
        openai_client = None
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Re-check previously skipped tweets for popularity increases"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Like count threshold for reconsidering tweets (default: 50)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Suppress verbose HTTP request logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logging.info("[*] Starting periodic recheck of skipped tweets")
    
    # Display current statistics
    try:
        stats = get_skipped_tweets_stats()
        if stats:
            logging.info(f"[*] Current stats: {stats['total_skipped_7d']} skipped (7d), "
                       f"{stats['recheck_candidates']} candidates ready for recheck")
            
            if stats['recheck_candidates'] == 0:
                logging.info("[*] No tweets ready for rechecking. Exiting.")
                return
                
        else:
            logging.warning("[*] Could not retrieve statistics")
            
    except Exception as e:
        logging.error(f"[!] Failed to get stats: {e}")
        return
    
    if args.dry_run:
        logging.info("[*] DRY RUN MODE - no changes will be made")
        return
    
    # Initialize Twitter scraper
    cookies_file = Path("twitter.cookies")
    if not cookies_file.exists():
        logging.error(f"[!] Twitter cookies file not found: {cookies_file}")
        sys.exit(1)
    
    try:
        scraper = Scraper(cookies=str(cookies_file))
        logging.info("[*] Twitter scraper initialized")
        
        # Run the recheck process
        recovered, still_skipped = recheck_skipped_tweets(
            scraper=scraper,
            popularity_threshold=args.threshold,
            apply_ai_safety_filter=openai_client is not None
        )
        
        # Report results
        if recovered > 0:
            logging.info(f"[✓] SUCCESS: Recovered {recovered} previously skipped tweets!")
        
        if still_skipped > 0:
            logging.info(f"[*] Checked {still_skipped} tweets that remain below threshold")
        
        if recovered == 0 and still_skipped == 0:
            logging.info("[*] No tweets were processed (possibly due to rate limits or errors)")
        
        logging.info("[✓] Periodic recheck completed")
        
    except Exception as e:
        logging.error(f"[!] Recheck failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 