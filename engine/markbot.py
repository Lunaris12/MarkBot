"""MarkBot: a chess bot that plays like me (GMMark12 on chess.com).

Its in UCI, which is the common language chess apps use to talk to engines, so I can
plug it into pretty much any chess GUI or put it on Lichess with lichess-bot.

When it's the bot's turn, it first checks whether we're still in my opening book. If we are, it
plays something I've actually played in that exact spot, and it picks between my options about as
often as I did. Once the game leaves my usual lines, it asks my trained model what I'd probably
play here. The model gives a chance for every legal move, and the bot picks one based on those
chances, so it won't play the same thing every game, just like how I don't.

It doesn't calculate or look ahead the way Stockfish does. It just plays the kind of move I'd
play including blunders because that's the whole point. To run, it needs three files in the
models/ folder: mark_model.onnx and moves.json from training/export_onnx.py, and book.json from
training/build_book.py.

You can tweak a few things in the chess app's engine settings. Rating sets how strong it plays
(1800 by default), and OpponentRating tells it how strong you are, since I play differently
against stronger players and so does the model. UseBook turns my opening book on or off.
Randomness controls how much it mixes things up: at 100 it varies its moves like I do, and at 0
it always plays my single most likely move. MinMoveProb (5% by default) makes it skip moves it
thinks I'd almost never play, so it doesn't randomly throw in something I'd never actually do.

On Lichess I don't have to set OpponentRating myself. lichess-bot tells the engine who it's playing
at the start of every game through a standard UCI setting called UCI_Opponent, and the bot uses that
person's real rating, so it plays a 1400 differently than a 2100 the same way I would.

Usage:
  python engine/markbot.py              
"""
import json
import os
import random
import sys

import chess
import numpy as np
import onnxruntime as ort

# When it's been packed into markbot.exe with PyInstaller, __file__ points at a temporary folder,
# so use the folder the .exe itself is in instead.
if getattr(sys, "frozen", False):
    HERE = os.path.dirname(os.path.abspath(sys.executable))
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
# Find the model files, first in the repo's models folder, otherwise right next to this file
MODELS_DIR = next(
    (d for d in (os.path.join(HERE, "..", "models"), HERE) if os.path.exists(os.path.join(d, "mark_model.onnx"))),
    os.path.join(HERE, "..", "models"),
)

# Maia-2 doesn't see exact ratings, it sees 100-point groups: under 1100, 1100-1199, ..., 2000 and up.
ELO_START, ELO_END = 1100, 2000


def rating_band(elo):
    if elo < ELO_START:
        return 0
    if elo >= ELO_END:
        return (ELO_END - ELO_START) // 100 + 1
    return (elo - ELO_START) // 100 + 1


def mirror_move(uci):
    """Flip a move upside down, e.g. e7e5 becomes e2e4. Used for undoing the board flip when I'm Black."""
    return uci[0] + str(9 - int(uci[1])) + uci[2] + str(9 - int(uci[3])) + uci[4:]


def encode(board):
    """Turn a board into numbers the model can read: 18 stacked 8x8 grids.

    Think of each grid as a see-through sheet over the board with 1s where something is:
      grids 0-5:   where White's pawns, knights, bishops, rooks, queens and king are
      grids 6-11:  the same for Black
      grid 12:     all 1s if it's White's turn
      grids 13-16: castling rights, one grid for each kind
      grid 17:     the en passant square, if there is one
    This has to match exactly how Maia-2 was trained (maia2's board_to_tensor), or it'd be reading garbage.
    """
    t = np.zeros((18, 8, 8), dtype=np.float32)
    for i, piece_type in enumerate(chess.PIECE_TYPES):
        for color, offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
            for square in board.pieces(piece_type, color):
                t[i + offset, square // 8, square % 8] = 1
    if board.turn == chess.WHITE:
        t[12] = 1
    rights = (board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE),
              board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK))
    for i, has_right in enumerate(rights):
        if has_right:
            t[13 + i] = 1
    if board.ep_square is not None:
        t[17, board.ep_square // 8, board.ep_square % 8] = 1
    return t


class MarkBot:
    def __init__(self):
        with open(os.path.join(MODELS_DIR, "moves.json"), encoding="utf-8") as f:
            self.moves = json.load(f)
        self.move_index = {m: i for i, m in enumerate(self.moves)}
        with open(os.path.join(MODELS_DIR, "book.json"), encoding="utf-8") as f:
            self.book = json.load(f)
        self.session = ort.InferenceSession(os.path.join(MODELS_DIR, "mark_model.onnx"),
                                            providers=["CPUExecutionProvider"])
        self.options = {"Rating": 1800, "OpponentRating": 1800, "UseBook": True,
                        "Randomness": 100, "MinMoveProb": 5}

    def move_chances(self, board):
        """Ask the model how likely I am to play each legal move here. Returns {move: chance}."""
        # The model was trained to always see things from White's side, so flip the board if it's Black's turn.
        flipped = board.turn == chess.BLACK
        b = board.mirror() if flipped else board
        legal = [m.uci() for m in b.legal_moves]
        scores = self.session.run(None, {
            "board": encode(b)[None],
            "elo_self": np.array([rating_band(self.options["Rating"])], dtype=np.int64),
            "elo_oppo": np.array([rating_band(self.options["OpponentRating"])], dtype=np.int64),
        })[0][0]
        # The model scores every move it knows, even impossible ones. Keep just the legal moves and
        # turn their scores into chances that add up to 100%.
        legal_scores = np.array([scores[self.move_index[m]] for m in legal], dtype=np.float64)
        chances = np.exp(legal_scores - legal_scores.max())
        chances /= chances.sum()
        return {(mirror_move(m) if flipped else m): float(p) for m, p in zip(legal, chances)}

    def choose(self, board):
        """Decide on a move. Returns the move plus a short note saying whether it came from the book or the model."""
        key = " ".join(board.fen().split()[:4])  # same way build_book.py labels positions
        if self.options["UseBook"] and key in self.book:
            options = {m: w for m, w in self.book[key].items() if chess.Move.from_uci(m) in board.legal_moves}
            if options:
                move = random.choices(list(options), weights=list(options.values()))[0]
                return move, "book"

        chances = self.move_chances(board)
        best = max(chances, key=chances.get)
        temperature = self.options["Randomness"] / 100
        if temperature <= 0:
            return best, f"model {chances[best]:.0%}"
        # Throw out moves I'd almost never play, then pick from what's left. Randomness under 100
        # leans harder toward my favourite moves, over 100 spreads the picks out more.
        floor = self.options["MinMoveProb"] / 100
        pool = {m: p ** (1 / temperature) for m, p in chances.items() if p >= floor} or {best: 1.0}
        move = random.choices(list(pool), weights=list(pool.values()))[0]
        return move, f"model {chances[move]:.0%}"


def uci_loop():
    """Listen for commands from the chess app, one line at a time, and answer each one."""
    bot, board = None, chess.Board()

    def send(line):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        command = parts[0]

        if command == "uci":  # the app asks what we're called and what settings we have
            send("id name MarkBot")
            send("id author GMMark12")
            send("option name Rating type spin default 1800 min 1000 max 2100")
            send("option name OpponentRating type spin default 1800 min 1000 max 2100")
            send("option name UseBook type check default true")
            send("option name Randomness type spin default 100 min 0 max 150")
            send("option name MinMoveProb type spin default 5 min 0 max 50")
            send("option name UCI_Opponent type string default none none human none")
            send("uciok")

        elif command == "isready":  # load everything now so the first move doesn't lag
            bot = bot or MarkBot()
            send("readyok")

        elif command == "setoption" and "name" in parts and "value" in parts:
            bot = bot or MarkBot()
            name = " ".join(parts[parts.index("name") + 1:parts.index("value")])
            value = " ".join(parts[parts.index("value") + 1:])
            if name == "UCI_Opponent":
                # lichess-bot sends who we're playing as "title, rating, human/computer, name", for
                # example "none 1650 human someone". If there's a rating given then play against that.
                info = value.split()
                if len(info) > 1 and info[1].isdigit():
                    bot.options["OpponentRating"] = int(info[1])
            elif name in bot.options:
                current = bot.options[name]
                bot.options[name] = (value.lower() == "true") if isinstance(current, bool) else int(value)

        elif command == "ucinewgame":
            board = chess.Board()

        elif command == "position":  # the app tells us the current position, for example "position startpos moves e2e4 e7e5"
            if parts[1] == "startpos":
                board, rest = chess.Board(), parts[2:]
            else:
                fen_end = parts.index("moves") if "moves" in parts else len(parts)
                board, rest = chess.Board(" ".join(parts[2:fen_end])), parts[fen_end:]
            if rest and rest[0] == "moves":
                for move in rest[1:]:
                    board.push_uci(move)

        elif command == "go":  # our turn, the clock doesn't matter because picking a move takes a few milliseconds
            bot = bot or MarkBot()
            move, note = bot.choose(board)
            send(f"info string {note}")
            send(f"bestmove {move}")

        elif command == "quit":
            break


if __name__ == "__main__":
    uci_loop()
