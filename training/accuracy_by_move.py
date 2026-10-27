"""Chart how often the bot plays my move at each point in the game.

evaluate.py only splits the test games into the first 12 moves and everything after, which hides
where exactly the bot stops being like me. This goes through the same 150 test games, with the same
book built without them, and works out the match rate for every group of 5 moves (moves 1 to 5,
6 to 10 and so on, with everything past move 40 lumped together since late moves are rare).
It does that for plain Maia-2, my fine-tuned model on its own, and the full bot with its book.

It saves the chart to results/accuracy_by_move.png and the numbers behind it to
results/accuracy_by_move.json.

Usage:
  python training/accuracy_by_move.py GMMark12
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # just save the picture, don't try to open a window
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_book import make_book  # noqa: E402
from evaluate import base_maia_picks, markbot, model_and_bot_picks, test_positions  # noqa: E402
from prep_data import load_games  # noqa: E402

GROUP = 5        # moves per point on the chart
LAST_GROUP = 41  # moves from here on all go in one final group
BOOK_MOVES = 12  # how far my opening book goes

# Colours: grey for plain Maia-2 since it's just the starting point, blue for the real bot. The model on
# its own is dashed and drawn last, because after the opening it picks the same moves as the full bot
# and a solid line would just hide behind it.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLES = {"Plain Maia-2": ("#8a8984", "-"), "MarkBot (book + model)": ("#2a78d6", "-"),
          "Fine-tuned model": ("#eb6834", "--")}


def group_of(ply):
    """Which group a position falls in, from the half-move number (my move 1 is ply 0 or 1)."""
    move = ply // 2 + 1
    return min((move - 1) // GROUP, (LAST_GROUP - 1) // GROUP)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user", help="chess.com username the bot is copying")
    ap.add_argument("--pgn", default="data/games.pgn")
    ap.add_argument("--holdout", type=int, default=150, help="the same test games prep_data.py set aside")
    ap.add_argument("--base-dir", default="maia2_models", help="where the Maia-2 base model is saved")
    ap.add_argument("--out", default="results/accuracy_by_move.png")
    args = ap.parse_args()

    games = load_games(args.pgn, args.user.lower())
    train_games, test_games = games[:-args.holdout], games[-args.holdout:]
    positions = test_positions(test_games)
    bot = markbot.MarkBot()
    bot.book = make_book(train_games)

    picks = {"Plain Maia-2": base_maia_picks(positions, args.base_dir)}
    picks["Fine-tuned model"], picks["MarkBot (book + model)"], _ = model_and_bot_picks(positions, bot)

    played = np.array([p["played"] for p in positions])
    groups = np.array([group_of(p["ply"]) for p in positions])
    n_groups = groups.max() + 1
    labels = [f"{g * GROUP + 1}-{g * GROUP + GROUP}" for g in range(n_groups - 1)] + [f"{LAST_GROUP}+"]
    counts = [int((groups == g).sum()) for g in range(n_groups)]
    rates = {}
    for name, p in picks.items():
        hit = np.array(p) == played
        rates[name] = [float(hit[groups == g].mean()) for g in range(n_groups)]

    print(f"{'moves':>8s} {'positions':>10s}" + "".join(f"{n[:16]:>18s}" for n in rates))
    for g in range(n_groups):
        print(f"{labels[g]:>8s} {counts[g]:>10d}" + "".join(f"{rates[n][g]:>18.1%}" for n in rates))

    # The chart.
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(n_groups)
    book_edge = BOOK_MOVES / GROUP - 0.5  # where move 12 falls between the groups
    ax.axvspan(-0.5, book_edge, color=GRID, alpha=0.45, lw=0)
    ax.text((book_edge - 0.5) / 2 + 0.25, 0.97, "my opening book", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color=MUTED)
    for name, (colour, dash) in STYLES.items():
        ax.plot(x, rates[name], color=colour, ls=dash, lw=2, marker="o", ms=5, label=name,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_xlabel("Move number", color=MUTED)
    ax.set_ylabel("Chance it plays my move", color=MUTED)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title("Rate that the bot plays my move", loc="left", color=INK, fontsize=12, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=INK,
              handlelength=2.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"moves": labels, "positions": counts, "match_rate": rates}, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
