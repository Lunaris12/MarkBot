"""Fine-tune Maia-2 on my games so it plays more like me.

Maia-2 already knows how a typical player at each rating moves. This keeps training it on the
examples from prep_data.py so it picks up my habits instead.

My games get split into three groups, oldest to newest:
  - training: most of my games. This is what the model learns from.
  - validation: the 100 games right before the test set. The model never trains on these;
    I use them to decide which saved version of the model is the best one.
  - test: my most recent games, set aside by prep_data.py. This script doesn't touch them,
    so the final evaluation is a fair check.

Every 250 steps it checks how often the model's top pick matches the move I really played in
the validation games, and saves the model whenever that's a new best.

With --only it trains on just some of my time controls, like --only rapid. That's for my experiment
on whether a bot trained on only my rapid games plays more like me than one trained on everything.
The validation games stay the same either way, so both versions get picked by the same yardstick.

Usage:
  python training/finetune.py
  python training/finetune.py --only rapid --out models/rapid_only.pt --log results/train_log_rapid_only.txt
  python training/finetune.py --epochs 2 --lr 3e-5 --out models/mark_model.pt
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from maia2 import model as maia2_model


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dataset.pt", help="output of prep_data.py")
    ap.add_argument("--out", default="models/mark_model.pt", help="where to save the best checkpoint")
    ap.add_argument("--log", default="results/train_log.txt")
    ap.add_argument("--base-dir", default="maia2_models", help="where the Maia-2 base weights are cached")
    ap.add_argument("--epochs", type=float, default=2.0, help="passes over the training data")
    ap.add_argument("--lr", type=float, default=3e-5, help="peak learning rate")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-games", type=int, default=100, help="games kept out of training to pick the best checkpoint")
    ap.add_argument("--eval-every", type=int, default=250, help="steps between validation checks")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", choices=["bullet", "blitz", "rapid", "daily"],
                    help="only train on these time controls, e.g. --only rapid")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    for path in (args.out, args.log):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    log = open(args.log, "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    # --- split the data -------------------------------------------------------
    d = torch.load(args.data, weights_only=False)
    not_test = ~d["is_test"]
    trainable_games = d["game"][not_test].unique().sort().values
    val_game_ids = trainable_games[-args.val_games:]
    is_val = not_test & torch.isin(d["game"], val_game_ids)
    train_idx = (not_test & ~is_val).nonzero().flatten()
    val_idx = is_val.nonzero().flatten()
    if args.only:
        # Drop every training position from a time control I didn't ask for.
        wanted = torch.tensor([c in args.only for c in d["cat"]])
        train_idx = train_idx[wanted[train_idx]]
        say(f"training on {', '.join(args.only)} games only")
    say(f"device {device} | train {len(train_idx)} positions | validation {len(val_idx)} positions "
        f"({args.val_games} games) | test {int(d['is_test'].sum())} positions (untouched)")

    def batch(idx):
        return (d["boards"][idx].float().to(device), d["elo_self"][idx].to(device), d["elo_oppo"][idx].to(device),
                d["legal"][idx].to(device), d["moves"][idx].to(device), d["weights"][idx].to(device))

    # --- model ----------------------------------------------------------------
    # Downloads Maia-2's rapid model the first time, then reuses the saved copy.
    net = maia2_model.from_pretrained("rapid", device=str(device), save_root=args.base_dir)

    @torch.no_grad()
    def validate():
        """How often the model's top legal move is the move I actually played, in the validation games."""
        net.eval()
        hits = 0
        for i in range(0, len(val_idx), 512):
            boards, elo_s, elo_o, legal, moves, _ = batch(val_idx[i:i + 512])
            logits = net(boards, elo_s, elo_o)[0].masked_fill(~legal, -1e9)
            hits += (logits.argmax(-1) == moves).sum().item()
        return hits / len(val_idx)

    best = validate()
    say(f"base Maia-2 validation match: {best:.3f}")
    torch.save(net.state_dict(), args.out)  # save the starting model too, so there's a file even if training never beats it

    # --- training -------------------------------------------------------------
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = int(args.epochs * len(train_idx) / args.batch_size)
    # Ramp the learning rate up over the first 5% of steps, then ease it back down,
    # so training doesn't destroy what Maia-2 already knows.
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    step, start = 0, time.time()
    while step < total_steps:
        order = train_idx[torch.randperm(len(train_idx))]
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            if step >= total_steps:
                break
            net.train()
            boards, elo_s, elo_o, legal, moves, weights = batch(order[i:i + args.batch_size])
            logits = net(boards, elo_s, elo_o)[0].masked_fill(~legal, -1e9)
            # How wrong the model was on each move, counted more for recent and rapid games.
            loss = (F.cross_entropy(logits, moves, reduction="none") * weights).sum() / weights.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1

            if step % 50 == 0:
                say(f"step {step}/{total_steps}  loss {loss.item():.3f}  {time.time() - start:.0f}s")
            if step % args.eval_every == 0 or step == total_steps:
                acc = validate()
                note = ""
                if acc > best:
                    best = acc
                    torch.save(net.state_dict(), args.out)
                    note = "  (new best, saved)"
                say(f"VALIDATION step {step}: match {acc:.3f}{note}")

    say(f"done in {time.time() - start:.0f}s | best validation match {best:.3f} | model saved to {args.out}")


if __name__ == "__main__":
    main()
