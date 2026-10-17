"""Build my opening book: what I actually play in the first 12 moves.

The neural net is decent at guessing my moves, but in the opening I have habits, so
there's no need to guess. This goes through my games, writes down what I played in each position,
and saves it so the bot can open the way I actually do. In a position I know well, the bot picks
one of my moves, choosing each about as often as I did. Once the game leaves my usual lines, the
neural net takes over.

A few rules so the book is really my opening repertoire:
  - Newer games count more, same as in prep_data.py (a game's weight halves every year,
    but never drops below 0.2).
  - A position only makes it in if I've reached it in at least 3 games.
  - A move only counts if I played it there at least twice, so a one time experiment
    or a mouse slip doesn't end up in my repertoire.
  - My 150 most recent games are left out by default. Those are the test games, and the
    book shouldn't get a look at them. Use --holdout 0 to build the final book
    from every game.

If I reach the same position through a different move order, it still counts as one position.

Usage:
  python training/build_book.py GMMark12
  python training/build_book.py GMMark12 --holdout 0 --out models/book.json
"""
import argparse
import collections
import datetime
import json
import os

import chess
import chess.pgn

RECENCY_FLOOR = 0.2


def position_key(board):
    """Describe a position by the board, whose turn it is, castling rights and en passant.

    Leaves out the move counters, so the same position always gets the same key.
    """
    return " ".join(board.fen().split()[:4])


def load_games(pgn_path, user):
    """All my normal chess games, oldest first. Same filters as prep_data.py."""
    games = []
    with open(pgn_path, encoding="utf-8") as f:
        while (g := chess.pgn.read_game(f)):
            h = g.headers
            if h.get("Variant", "Standard").lower() not in ("standard", "") or h.get("SetUp") == "1":
                continue
            white = h.get("White", "").lower() == user
            if not white and h.get("Black", "").lower() != user:
                continue
            date = datetime.date(*map(int, h.get("UTCDate", h["Date"]).split(".")))
            games.append((date, h.get("UTCTime", ""), g, white))
    games.sort(key=lambda x: (x[0], x[1]))
    return games


def describe(book, board):
    """What I play in a position, written like 'e4 70%, d4 30%', for the printout at the end."""
    moves = book.get(position_key(board))
    if not moves:
        return "(not in book)"
    total = sum(moves.values())
    ranked = sorted(moves.items(), key=lambda kv: -kv[1])
    def pct(w):
        p = 100 * w / total
        return "<1%" if p < 0.5 else f"{p:.0f}%" # Shows as 0% otherwise which is misleading
    return ", ".join(f"{board.san(chess.Move.from_uci(m))} {pct(w)}" for m, w in ranked)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user", help="chess.com username whose openings to record")
    ap.add_argument("--pgn", default="data/games.pgn")
    ap.add_argument("--out", default="models/book.json")
    ap.add_argument("--holdout", type=int, default=150, help="most recent games to leave out (0 = use all)")
    ap.add_argument("--max-moves", type=int, default=12, help="how many of my moves per game to record")
    ap.add_argument("--min-games", type=int, default=3, help="games needed before a position enters the book")
    ap.add_argument("--min-plays", type=int, default=2, help="times I must have played a move for it to count")
    args = ap.parse_args()

    games = load_games(args.pgn, args.user.lower())
    if args.holdout:
        games = games[:-args.holdout]
    if not games:
        raise SystemExit("No games left to build a book from.")
    latest = games[-1][0]

    # For each position, and each move I played there, counts how many times, and the total weight
    counts = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for date, _, g, white in games:
        age_years = (latest - date).days / 365
        weight = max(0.5 ** age_years, RECENCY_FLOOR)
        board = g.board()
        for ply, move in enumerate(g.mainline_moves()):
            if ply >= 2 * args.max_moves:
                break
            if board.turn == (chess.WHITE if white else chess.BLACK):
                entry = counts[position_key(board)][move.uci()]
                entry[0] += 1
                entry[1] += weight
            board.push(move)

    book = {}
    for pos, moves in counts.items():
        if sum(n for n, _ in moves.values()) < args.min_games:
            continue
        kept = {m: round(w, 3) for m, (n, w) in moves.items() if n >= args.min_plays}
        if kept:
            book[pos] = kept

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(book, f, separators=(",", ":"))

    print(f"{len(book)} positions in the book, from {len(games)} games -> {args.out}")
    start = chess.Board()
    print(f"  as White, move 1:   {describe(book, start)}")
    for first in ("e4", "d4"):
        b = chess.Board()
        b.push_san(first)
        print(f"  as Black vs 1.{first}: {describe(book, b)}")


if __name__ == "__main__":
    main()
