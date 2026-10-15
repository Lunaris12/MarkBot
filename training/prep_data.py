"""Turn a player's PGN into training data for fine-tuning Maia-2.

Each training example is one position where the player was to move:
  - the board, encoded the way Maia-2 expects
  - the move the player actually made
  - the player's rating and the opponent's rating in that game
  - a weight saying how much this example should count in training

Weighting:
  - Recent games count more: weight halves for every year of age, never below 0.2.
  - Rapid games count 2x, bullet 0.5x, since I want the bot to play like my rapid self.
  - Moves made with under 10 seconds on the clock are dropped (moves under time pressure aren't my style).

The most recent --holdout games are marked as test data and must never be trained on.
They are used later to measure how often the bot picks the move I really played.

Usage:
  python training/prep_data.py GMMark12
  python training/prep_data.py GMMark12 --pgn data/games.pgn --out data/dataset.pt --holdout 150
"""
import argparse
import collections
import datetime
import re

import chess
import chess.pgn
import torch
from maia2.utils import board_to_tensor, create_elo_dict, get_all_possible_moves, map_to_category, mirror_move

TIME_CONTROL_WEIGHT = {"rapid": 2.0, "blitz": 1.0, "bullet": 0.5, "daily": 1.0}
MIN_CLOCK_SECONDS = 10
RECENCY_FLOOR = 0.2


def time_category(tc):
    """chess.com TimeControl header -> bullet / blitz / rapid / daily."""
    if "/" in tc:  # daily games look like "1/259200"
        return "daily"
    base = int(tc.split("+")[0])
    return "bullet" if base < 180 else "blitz" if base < 600 else "rapid"


def clock_seconds(comment):
    """Read the remaining clock from a move comment like {[%clk 0:02:31.5]}."""
    m = re.search(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]", comment)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None


def load_games(pgn_path, user):
    """All standard-chess games involving the user, oldest first."""
    games = []
    with open(pgn_path, encoding="utf-8") as f:
        while (g := chess.pgn.read_game(f)):
            h = g.headers
            if h.get("Variant", "Standard").lower() not in ("standard", "") or h.get("SetUp") == "1":
                continue  # skip Chess960 and games from custom positions
            white = h.get("White", "").lower() == user
            if not white and h.get("Black", "").lower() != user:
                continue
            date = datetime.date(*map(int, h.get("UTCDate", h["Date"]).split(".")))
            games.append((date, h.get("UTCTime", ""), g, white))
    games.sort(key=lambda x: (x[0], x[1]))
    return games


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user", help="chess.com username whose moves to learn")
    ap.add_argument("--pgn", default="data/games.pgn")
    ap.add_argument("--out", default="data/dataset.pt")
    ap.add_argument("--holdout", type=int, default=150, help="most recent games kept out of training")
    args = ap.parse_args()

    all_moves = get_all_possible_moves()
    move_index = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    games = load_games(args.pgn, args.user.lower())
    if len(games) <= args.holdout:
        raise SystemExit(f"Only {len(games)} games found; need more than --holdout ({args.holdout}).")
    latest = games[-1][0]
    first_test_game = len(games) - args.holdout

    cols = collections.defaultdict(list)
    stats = collections.Counter()
    for gi, (date, _, g, white) in enumerate(games):
        h = g.headers
        cat = time_category(h["TimeControl"])
        my_elo, opp_elo = (int(h["WhiteElo"]), int(h["BlackElo"])) if white else (int(h["BlackElo"]), int(h["WhiteElo"]))
        age_years = (latest - date).days / 365
        weight = max(0.5 ** age_years, RECENCY_FLOOR) * TIME_CONTROL_WEIGHT[cat]

        board = g.board()
        for ply, node in enumerate(g.mainline()):
            if board.turn == (chess.WHITE if white else chess.BLACK):
                clk = clock_seconds(node.comment)
                if cat != "daily" and clk is not None and clk < MIN_CLOCK_SECONDS:
                    stats["dropped_low_clock"] += 1
                else:
                    # Maia-2 always sees the board from White's side, so flip when I was Black.
                    b = board if board.turn == chess.WHITE else board.mirror()
                    uci = node.move.uci() if board.turn == chess.WHITE else mirror_move(node.move.uci())
                    legal = torch.zeros(len(all_moves), dtype=torch.bool)
                    legal[[move_index[m.uci()] for m in b.legal_moves]] = True
                    cols["boards"].append(board_to_tensor(b).to(torch.uint8))
                    cols["elo_self"].append(map_to_category(my_elo, elo_dict))
                    cols["elo_oppo"].append(map_to_category(opp_elo, elo_dict))
                    cols["moves"].append(move_index[uci])
                    cols["weights"].append(weight)
                    cols["legal"].append(legal)
                    cols["ply"].append(ply)
                    cols["game"].append(gi)
                    cols["cat"].append(cat)
                    stats["kept"] += 1
            board.push(node.move)

    is_test = torch.tensor([gi >= first_test_game for gi in cols["game"]])
    data = dict(
        boards=torch.stack(cols["boards"]),
        elo_self=torch.tensor(cols["elo_self"]),
        elo_oppo=torch.tensor(cols["elo_oppo"]),
        moves=torch.tensor(cols["moves"]),
        weights=torch.tensor(cols["weights"], dtype=torch.float32),
        legal=torch.stack(cols["legal"]),
        ply=torch.tensor(cols["ply"]),
        game=torch.tensor(cols["game"]),
        is_test=is_test,
        cat=cols["cat"],
    )
    torch.save(data, args.out)

    train = ~is_test
    print(f"{len(games)} games -> {stats['kept']} positions ({stats['dropped_low_clock']} low-clock moves dropped)")
    print(f"train: {int(train.sum())} positions, test: {int(is_test.sum())} positions "
          f"(test games {games[first_test_game][0]} to {latest})")
    total_weight = data["weights"][train].sum()
    for c in ("blitz", "rapid", "bullet", "daily"):
        sel = torch.tensor([x == c for x in cols["cat"]]) & train
        print(f"  {c:6s} {int(sel.sum()):6d} positions, {100 * data['weights'][sel].sum() / total_weight:3.0f}% of training weight")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
