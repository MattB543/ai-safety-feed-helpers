#!/usr/bin/env python3
"""Fetch the AISafetyFeed following list via twscrape."""
import asyncio
import json
import re

# Monkey-patch twscrape for Twitter's malformed JSON (Dec 2025+)
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


async def main():
    api = API("accounts.db")
    await api.pool.reset_locks()
    print("Locks cleared")
    stats = await api.pool.stats()
    print(f"Pool: {stats}")

    # Get AISafetyFeed user
    user = await api.user_by_login("AISafetyFeed")
    print(f"User: @{user.username} (ID: {user.id}, following: {user.friendsCount}, followers: {user.followersCount})")

    # Get following list
    count = 0
    following = []
    async for u in api.following(user.id, limit=500):
        following.append(u.username)
        count += 1
        if count % 50 == 0:
            print(f"  ...fetched {count}")

    print(f"\nTotal following: {count}")
    for name in sorted(following):
        print(f"  @{name}")

asyncio.run(main())
