"""Download every chess.com game for a player into one PGN file.

Uses chess.com's public API (no login needed):
  https://api.chess.com/pub/player/<user>/games/archives  -> list of monthly archives
  <archive>/pgn                                          -> that month's games as PGN

Usage:
  python training/fetch_games.py GMMark12
  python training/fetch_games.py GMMark12 --out data/games.pgn --since 2024-01
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.chess.com/pub/player/{user}/games/archives"
# chess.com asks API clients to send a descriptive User-Agent.
HEADERS = {"User-Agent": "MarkBot game fetcher (personal chess-bot project)"}


def get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:  # rate limited: back off
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user", help="chess.com username")
    ap.add_argument("--out", default="data/games.pgn", help="output PGN path (default: data/games.pgn)")
    ap.add_argument("--since", help="only months from YYYY-MM onward")
    args = ap.parse_args()

    user = args.user.lower()
    try:
        archives = json.loads(get(API.format(user=user)))["archives"]
    except urllib.error.HTTPError as e:
        sys.exit(f"Could not get archives for '{user}' (HTTP {e.code}). Check the username.")
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach chess.com ({e.reason}). Check your internet connection.")

    if args.since:
        archives = [a for a in archives if "/".join(a.rsplit("/", 2)[-2:]) >= args.since.replace("-", "/")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for url in archives:
            month = "/".join(url.rsplit("/", 2)[-2:])
            pgn = get(url + "/pgn").strip()
            n = pgn.count("[Event ")
            total += n
            print(f"{month}: {n} games")
            if pgn:
                f.write(pgn + "\n\n")
            time.sleep(0.2)  # be polite to the API: requests one at a time
    print(f"\nSaved {total} games from {len(archives)} months to {args.out}")


if __name__ == "__main__":
    main()
