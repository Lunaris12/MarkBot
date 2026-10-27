"""Check how much the bot actually plays like me, on games it has never seen.

My 150 most recent games were set aside back in prep_data.py and never used for training, so
they're a fair test. For every position in those games where it was my turn, this asks three
different players what they'd play and checks whether their top choice is the move I really made.
First, plain Maia-2, the model I started from, which plays like a typical human at my rating.
Then my fine-tuned model on its own. Finally, the full bot, which uses my opening book first 
and the fine-tuned model after that

The full bot here uses a book built without the test games, so it can't cheat by having seen
them. Moves I made with under 10 seconds left are skipped, same as in training. Each player is
told the real ratings from that game, so it's the same situation I was in.

It reports the match rate overall, for the first 12 moves which is what my opening book covers, and for everything after, then saves
the numbers to results/eval_results.json. Nobody repeats the same move in the same spot every
time, so 100% isn't possible so what matters is how much better the bot does than plain Maia-2.
It also splits the match rate up by time control, so I can see if it copies my rapid games
better than my blitz or bullet ones.

--model tests a different model file instead of my normal one, which is how I compare the
rapid-only version from my experiment against the one trained on all my games.

Usage:
  python training/evaluate.py GMMark12
  python training/evaluate.py GMMark12 --model models/rapid_only.onnx --out results/eval_rapid_only.json
"""
import argparse
import json
import os
import sys

import chess
import numpy as np
import onnxruntime as ort
import torch
from maia2 import model as maia2_model
from maia2.utils import board_to_tensor, create_elo_dict, get_all_possible_moves, map_to_category, mirror_move

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from build_book import make_book, position_key  
from prep_data import MIN_CLOCK_SECONDS, clock_seconds, load_games, time_category  
import markbot  

OPENING_PLIES = 24  # the first 12 moves for each side, same as the opening book


def test_positions(games):
    """Every position in the test games where it was my turn, with what I played and both ratings."""
    positions = []
    for _, _, g, white in games:
        h = g.headers
        cat = time_category(h["TimeControl"])
        my_elo, opp_elo = (int(h["WhiteElo"]), int(h["BlackElo"])) if white else (int(h["BlackElo"]), int(h["WhiteElo"]))
        board = g.board()
        for ply, node in enumerate(g.mainline()):
            if board.turn == (chess.WHITE if white else chess.BLACK):
                clk = clock_seconds(node.comment)
                if not (cat != "daily" and clk is not None and clk < MIN_CLOCK_SECONDS):
                    positions.append(dict(board=board.copy(), played=node.move.uci(), ply=ply,
                                          my_elo=my_elo, opp_elo=opp_elo, cat=cat))
            board.push(node.move)
    return positions


@torch.no_grad()
def base_maia_picks(positions, base_dir):
    """Plain Maia-2's top move in each position, run in batches with PyTorch."""
    net = maia2_model.from_pretrained("rapid", device="cpu", save_root=base_dir).eval()
    all_moves = get_all_possible_moves()
    move_index = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    picks = []
    for start in range(0, len(positions), 512):
        chunk = positions[start:start + 512]
        boards, legal, elo_s, elo_o, flipped = [], [], [], [], []
        for p in chunk:
            # Same flip as in training since Maia-2 always sees the board from White's side.
            flip = p["board"].turn == chess.BLACK
            b = p["board"].mirror() if flip else p["board"]
            mask = torch.zeros(len(all_moves), dtype=torch.bool)
            mask[[move_index[m.uci()] for m in b.legal_moves]] = True
            boards.append(board_to_tensor(b))
            legal.append(mask)
            elo_s.append(map_to_category(p["my_elo"], elo_dict))
            elo_o.append(map_to_category(p["opp_elo"], elo_dict))
            flipped.append(flip)
        scores = net(torch.stack(boards), torch.tensor(elo_s), torch.tensor(elo_o))[0]
        best = scores.masked_fill(~torch.stack(legal), -1e9).argmax(-1).tolist()
        for idx, flip in zip(best, flipped):
            move = all_moves[idx]
            picks.append(mirror_move(move) if flip else move)
    return picks


def model_and_bot_picks(positions, bot):
    """The fine-tuned model's top move on its own, and the full bot's top move (book first), in each position."""
    model_picks, bot_picks, from_book = [], [], 0
    for p in positions:
        bot.options["Rating"], bot.options["OpponentRating"] = p["my_elo"], p["opp_elo"]
        chances = bot.move_chances(p["board"])
        model_pick = max(chances, key=chances.get)
        model_picks.append(model_pick)
        book_moves = bot.book.get(position_key(p["board"]))
        if book_moves:
            # The bot's most likely book move is the one I played most in that spot.
            bot_picks.append(max(book_moves, key=book_moves.get))
            from_book += 1
        else:
            bot_picks.append(model_pick)
    return model_picks, bot_picks, from_book


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user", help="chess.com username the bot is copying")
    ap.add_argument("--pgn", default="data/games.pgn")
    ap.add_argument("--holdout", type=int, default=150, help="the same test games prep_data.py set aside")
    ap.add_argument("--base-dir", default="maia2_models", help="where the Maia-2 base model is saved")
    ap.add_argument("--model", default="models/mark_model.onnx", help="which fine-tuned model to test")
    ap.add_argument("--out", default="results/eval_results.json")
    args = ap.parse_args()

    games = load_games(args.pgn, args.user.lower())
    train_games, test_games = games[:-args.holdout], games[-args.holdout:]
    positions = test_positions(test_games)
    print(f"{len(test_games)} test games ({test_games[0][0]} to {test_games[-1][0]}), {len(positions)} positions")

    # The bot but with a book that hasn't seen the test games.
    bot = markbot.MarkBot()
    bot.book = make_book(train_games)
    bot.session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    base_picks = base_maia_picks(positions, args.base_dir)
    model_picks, bot_picks, from_book = model_and_bot_picks(positions, bot)

    played = [p["played"] for p in positions]
    opening = np.array([p["ply"] < OPENING_PLIES for p in positions])
    cats = np.array([p["cat"] for p in positions])
    # Daily games are left out of the split since I've barely played any.
    tcs = [c for c in ("rapid", "blitz", "bullet") if (cats == c).any()]

    def match_rates(picks):
        hit = np.array([a == b for a, b in zip(picks, played)])
        return {"all": float(hit.mean()), "first_12_moves": float(hit[opening].mean()),
                "after_move_12": float(hit[~opening].mean()),
                "by_time_control": {c: float(hit[cats == c].mean()) for c in tcs}}

    results = {
        "model": args.model,
        "test_games": len(test_games),
        "positions": len(positions),
        "positions_from_book": from_book,
        "Base Maia-2": match_rates(base_picks),
        "Fine-tuned model": match_rates(model_picks),
        "MarkBot (book + model)": match_rates(bot_picks),
    }

    # Table of results
    print(f"\n{'':24s} {'all moves':>10s} {'first 12':>10s} {'after 12':>10s}")
    for name in ("Base Maia-2", "Fine-tuned model", "MarkBot (book + model)"):
        r = results[name]
        print(f"{name:24s} {r['all']:>10.1%} {r['first_12_moves']:>10.1%} {r['after_move_12']:>10.1%}")
    print(f"\n{'by time control':24s}" + "".join(f"{c:>10s}" for c in tcs))
    for name in ("Base Maia-2", "Fine-tuned model", "MarkBot (book + model)"):
        by = results[name]["by_time_control"]
        print(f"{name:24s}" + "".join(f"{by[c]:>10.1%}" for c in tcs))
    print("positions" + " " * 15 + "".join(f"{int((cats == c).sum()):>10d}" for c in tcs))
    print(f"\nthe book answered {from_book} of {len(positions)} positions")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
