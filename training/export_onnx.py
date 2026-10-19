"""Turn my trained model into a file the bot can run without PyTorch.

PyTorch is great for training but too big to bring along just to play chess. ONNX is a
standard file format for trained models, and a small library called onnxruntime can run it,
so the bot stays light and quick to install.

This loads the model I trained with finetune.py and keeps only the part that picks moves. Maia-2
also guesses things like who's winning, but the bot doesn't need any of that. It saves the result as
models/mark_model.onnx, then double-checks that the new file gives the same answers as the original,
so I know nothing broke in the conversion. Last, it saves Maia-2's list of every move it knows
into models/moves.json. The model answers with a number for each move on that list, so the bot needs
it to know which number is which move.

Usage:
  python training/export_onnx.py
  python training/export_onnx.py --model models/mark_model.pt --out models/mark_model.onnx
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch
from maia2 import model as maia2_model
from maia2.utils import get_all_possible_moves


class MovePredictor(torch.nn.Module):
    """A thin wrapper around Maia-2 that only hands back the move scores.

    Maia-2 gives back three things at once. The bot only cares about the first which is a score for every move.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, board, elo_self, elo_oppo):
        return self.net(board, elo_self, elo_oppo)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/mark_model.pt", help="the model finetune.py saved")
    ap.add_argument("--out", default="models/mark_model.onnx")
    ap.add_argument("--base-dir", default="maia2_models", help="where the Maia-2 base model is saved")
    args = ap.parse_args()

    # Start from a fresh Maia-2, then load in everything my training changed.
    net = maia2_model.from_pretrained("rapid", device="cpu", save_root=args.base_dir)
    net.load_state_dict(torch.load(args.model, map_location="cpu"))
    predictor = MovePredictor(net).eval()

    # The export works by running the model once and recording what it does, so it needs some input.
    # Two random fake positions are fine, it only cares about the shape of the input, not what's in it.
    example = (torch.rand(2, 18, 8, 8).round(), torch.tensor([8, 8]), torch.tensor([8, 7]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        predictor, example, args.out,
        input_names=["board", "elo_self", "elo_oppo"],
        output_names=["logits"],
        # Don't lock it to exactly 2 positions as the engine sends 1 at a time.
        dynamic_axes={name: {0: "n"} for name in ["board", "elo_self", "elo_oppo", "logits"]},
        opset_version=17,
        dynamo=False,
    )

    # Run the same fake positions through both versions. If the scores don't match, something went wrong.
    with torch.no_grad():
        expected = predictor(*example).numpy()
    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = session.run(None, {"board": example[0].numpy(), "elo_self": example[1].numpy(),
                             "elo_oppo": example[2].numpy()})[0]
    diff = float(np.abs(expected - got).max())
    if diff > 1e-3:
        raise SystemExit(f"ONNX and PyTorch disagree (biggest difference {diff:.4f}); the export went wrong.")

    moves_path = os.path.join(os.path.dirname(args.out) or ".", "moves.json")
    with open(moves_path, "w", encoding="utf-8") as f:
        json.dump(get_all_possible_moves(), f)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"saved {args.out} ({size_mb:.0f} MB) and {moves_path}")
    print(f"ONNX matches PyTorch (biggest difference {diff:.1e})")


if __name__ == "__main__":
    main()
