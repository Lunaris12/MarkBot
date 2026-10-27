"""Retrain the bot on my newest games once a month, and only switch to the new model if it's actually better.

Windows Task Scheduler runs this on the 1st of every month through windows/monthly_retrain.bat, and
windows/schedule_retrain.bat sets that up. It can also just be run by hand.

First it downloads all my games again. If there's nothing new since last time it stops right there,
since retraining on the same games would just give me the same bot. Otherwise it goes through the
whole pipeline but saves the new model and opening book into models/candidate instead of on top of
the real ones, so the bot that's playing never gets touched halfway through. Its training log and
test results go in there too, so I can look at them even if it loses.

Then both the new model and the one the bot is using right now get tested on my 150 most recent
games. The new games land at the end of that test set, and the older part of it was already the
test set last time, so the current model has never trained on any of those games either and the
comparison is fair. If the new model matches my moves at least as often, it replaces the old one,
which gets kept in models/previous in case I ever want it back. If it's worse, the old one stays.

When the new model wins and deploy/server.txt exists, it also copies the new files to my Google
server and runs update_server.sh there so the Lichess bot switches over too. That file is one line
with the server login and zone, like myname@my-server us-east1-b, and it's
in .gitignore so my server details stay off GitHub. It needs the Google Cloud CLI (gcloud)
installed and logged in on my PC. Every run adds a line to results/retrain_history.txt.

Usage:
  python training/monthly_retrain.py GMMark12
  python training/monthly_retrain.py GMMark12 --check-server
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

PY = sys.executable
FILES = ["mark_model.onnx", "moves.json", "book.json", "mark_model.pt"]
SERVER_FILES = ["mark_model.onnx", "moves.json", "book.json"]


def run(*args):
    """Run one of my training scripts and stop everything if it fails."""
    print(f"\n>>> {' '.join(args)}", flush=True)
    subprocess.run([PY, *args], check=True)


def count_games(path):
    """Every game in a PGN starts with an [Event tag, so counting those counts the games."""
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return f.read().count("[Event ")


def match_rate(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["MarkBot (book + model)"]["all"]


def server_settings():
    """The login and zone from deploy/server.txt, or None if I haven't set a server up."""
    if not os.path.exists("deploy/server.txt"):
        return None
    login, zone = open("deploy/server.txt", encoding="utf-8").read().split()[:2]
    return login, zone


def gcloud(*args):
    exe = shutil.which("gcloud")
    if not exe:
        raise SystemExit("gcloud wasn't found. Install the Google Cloud CLI and run gcloud init first.")
    print(f"\n>>> gcloud {' '.join(args)}", flush=True)
    subprocess.run([exe, *args], check=True)


def update_server(login, zone):
    """Copy the new model files to the server and restart the bot there."""
    instance = login.split("@")[1]
    gcloud("compute", "scp", *[f"models/{f}" for f in SERVER_FILES], f"{login}:MarkBot/models/", "--zone", zone)
    gcloud("compute", "ssh", login, "--zone", zone, "--command", "bash MarkBot/deploy/update_server.sh")
    print(f"{instance} is running the new model.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user", help="chess.com username to retrain on")
    ap.add_argument("--epochs", default="2", help="passed on to finetune.py")
    ap.add_argument("--check-server", action="store_true",
                    help="just test that gcloud can reach the server, then stop")
    args = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    server = server_settings()
    if args.check_server:
        if not server:
            raise SystemExit("deploy/server.txt doesn't exist yet, so there's no server to check.")
        gcloud("compute", "ssh", server[0], "--zone", server[1], "--command", "ls MarkBot/models")
        print("The server is reachable, so new models will get sent there automatically.")
        return

    today = datetime.date.today().isoformat()
    old_count = count_games("data/games.pgn")
    run("training/fetch_games.py", args.user, "--out", "data/games_new.pgn")
    new_count = count_games("data/games_new.pgn")
    if new_count <= old_count:
        os.remove("data/games_new.pgn")
        line = f"{today}  {old_count} games, nothing new so nothing was retrained"
    else:
        os.replace("data/games_new.pgn", "data/games.pgn")
        os.makedirs("models/candidate", exist_ok=True)
        run("training/prep_data.py", args.user)
        run("training/finetune.py", "--epochs", args.epochs, "--out", "models/candidate/mark_model.pt",
            "--log", "models/candidate/train_log.txt")
        run("training/export_onnx.py", "--model", "models/candidate/mark_model.pt",
            "--out", "models/candidate/mark_model.onnx")
        run("training/build_book.py", args.user, "--holdout", "0", "--out", "models/candidate/book.json")
        # Same test games for both, the current model first and then the new one.
        run("training/evaluate.py", args.user, "--out", "models/candidate/eval_current.json")
        run("training/evaluate.py", args.user, "--model", "models/candidate/mark_model.onnx",
            "--out", "models/candidate/eval.json")

        current, new = match_rate("models/candidate/eval_current.json"), match_rate("models/candidate/eval.json")
        line = f"{today}  {new_count} games (+{new_count - old_count})  current {current:.1%}  new {new:.1%}"
        if new >= current:
            os.makedirs("models/previous", exist_ok=True)
            for f in FILES:
                if os.path.exists(f"models/{f}"):
                    shutil.copy2(f"models/{f}", f"models/previous/{f}")
                shutil.copy2(f"models/candidate/{f}", f"models/{f}")
            shutil.copy2("models/candidate/eval.json", "results/eval_results.json")
            shutil.copy2("models/candidate/train_log.txt", "results/train_log.txt")
            line += "  -> switched to the new model"
            if server:
                update_server(*server)
                line += ", server updated"
            else:
                line += ", no server set up so copy the files over by hand"
        else:
            line += "  -> kept the current model"

    print("\n" + line)
    with open("results/retrain_history.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
