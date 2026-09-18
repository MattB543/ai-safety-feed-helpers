#!/usr/bin/env python3
"""
setup_twitter_accounts.py
─────────────────────────
Setup twscrape accounts database from accounts.txt file.

This script reads Twitter account credentials from accounts.txt and adds them
to twscrape's SQLite database for use by the scraper.

accounts.txt format (one account per line):
    username:password:email:email_password:auth_token=XXX; ct0=YYY

Usage:
    python setup_twitter_accounts.py
    python setup_twitter_accounts.py --accounts-file custom_accounts.txt
    python setup_twitter_accounts.py --check  # Just check status, don't add
"""

import asyncio
import argparse
import sys
from pathlib import Path

try:
    from twscrape import API, AccountsPool
except ImportError:
    print("ERROR: twscrape not installed. Run: pip install twscrape")
    sys.exit(1)

# ───── Configuration ──────────────────────────────────────────────────────────
DEFAULT_ACCOUNTS_FILE = "accounts.txt"
DEFAULT_DB_PATH = "accounts.db"


def parse_accounts_file(file_path: Path) -> list[dict]:
    """
    Parse accounts.txt file and return list of account dictionaries.

    Expected format: username:password:email:email_password:cookies
    Where cookies is: auth_token=XXX; ct0=YYY
    """
    accounts = []

    if not file_path.exists():
        print(f"ERROR: Accounts file not found: {file_path}")
        return accounts

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            parts = line.split(":")

            # Expected: username:password:email:email_password:cookies
            # But cookies contain ":" so we need at least 5 parts
            if len(parts) < 5:
                print(f"WARNING: Line {line_num}: Invalid format, skipping: {line[:50]}...")
                continue

            username = parts[0]
            password = parts[1]
            email = parts[2]
            email_password = parts[3]
            # Cookies are everything after the 4th colon
            cookies = ":".join(parts[4:])

            accounts.append({
                "username": username,
                "password": password,
                "email": email,
                "email_password": email_password,
                "cookies": cookies.strip()
            })

    return accounts


async def setup_accounts(
    accounts_file: Path,
    db_path: str,
    force: bool = False
):
    """Add accounts from file to twscrape database."""

    print("=" * 60)
    print("twscrape Account Setup")
    print("=" * 60)

    # Parse accounts file
    print(f"\n[*] Reading accounts from: {accounts_file}")
    accounts = parse_accounts_file(accounts_file)

    if not accounts:
        print("ERROR: No valid accounts found in file.")
        return

    print(f"   Found {len(accounts)} account(s)")

    # Initialize API
    print(f"\n[*] Using database: {db_path}")
    api = API(db_path)

    # Check existing accounts
    existing = await api.pool.accounts_info()
    existing_usernames = {(acc.username if hasattr(acc, 'username') else acc.get('username', '')) for acc in existing}
    print(f"   Existing accounts in DB: {len(existing_usernames)}")

    # Add accounts
    added = 0
    skipped = 0

    for acc in accounts:
        username = acc["username"]

        if username in existing_usernames and not force:
            print(f"   [-] Skipping {username} (already exists, use --force to re-add)")
            skipped += 1
            continue

        try:
            print(f"   [+] Adding {username}...")

            # Add account with cookies
            await api.pool.add_account(
                username=acc["username"],
                password=acc["password"],
                email=acc["email"],
                email_password=acc["email_password"],
                cookies=acc["cookies"]
            )
            added += 1
            print(f"       OK - Added successfully")

        except Exception as e:
            print(f"       FAIL - Failed to add {username}: {e}")

    print(f"\n[*] Summary: {added} added, {skipped} skipped")

    # Show final status
    await check_accounts(db_path)


async def check_accounts(db_path: str):
    """Check status of accounts in the database."""

    print("\n" + "-" * 60)
    print("Account Status")
    print("-" * 60)

    api = API(db_path)

    try:
        accounts = await api.pool.accounts_info()

        if not accounts:
            print("   No accounts in database.")
            return

        for acc in accounts:
            if hasattr(acc, 'username'):
                uname, active, logged, err = acc.username, acc.active, acc.logged_in, acc.error_msg
            else:
                uname, active, logged, err = acc.get('username','?'), acc.get('active',False), acc.get('logged_in',False), acc.get('error_msg')
            status = "Active" if active else "Inactive"
            li = "logged in" if logged else "NOT logged in"
            print(f"   @{uname}: {status}, {li}")
            if err:
                print(f"      Error: {err}")

        # Get pool stats
        stats = await api.pool.stats()
        print(f"\n   Pool stats: {stats}")

    except Exception as e:
        print(f"   Error checking accounts: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Setup twscrape accounts from accounts.txt"
    )
    parser.add_argument(
        "--accounts-file", "-a",
        type=Path,
        default=Path(DEFAULT_ACCOUNTS_FILE),
        help=f"Path to accounts file (default: {DEFAULT_ACCOUNTS_FILE})"
    )
    parser.add_argument(
        "--db", "-d",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to twscrape database (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Only check account status, don't add new accounts"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-add accounts that already exist"
    )

    args = parser.parse_args()

    if args.check:
        await check_accounts(args.db)
    else:
        await setup_accounts(args.accounts_file, args.db, args.force)


if __name__ == "__main__":
    asyncio.run(main())
