#!/usr/bin/env bash
# This sets MarkBot up on a Linux server so it can stay on Lichess all the time without my PC being on.
# I wrote it for Ubuntu, like the free servers from Oracle or Google.
#
# Before running it, clone this repo on the server and copy my three model files into MarkBot/models,
# since they're too big for GitHub and aren't in the repo. Then run bash deploy/setup_server.sh and it
# installs everything, asks for the bot account's token once and sets MarkBot up as a service. The service
# starts the bot whenever the server boots and restarts it if it ever crashes. Running it again later is
# safe, it skips whatever is already done and just reinstalls the service.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

# The model files have to be here first or the bot has nothing to play with.
for f in mark_model.onnx moves.json book.json; do
  if [ ! -f "models/$f" ]; then
    echo "models/$f is missing. Copy the model files over from your PC first, the README shows how."
    exit 1
  fi
done

echo "Installing Python and Git..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git >/dev/null

# The server only plays, so it just needs the small engine packages and not PyTorch.
echo "Setting up the Python environment..."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r engine/requirements.txt

echo "Getting lichess-bot..."
[ -d lichess-bot ] || git clone -q --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git lichess-bot
.venv/bin/pip install -q -r lichess-bot/requirements.txt

# There's no markbot.exe on Linux, so the config switches to running markbot.py with the Python from .venv. 
if [ ! -f lichess-bot/config.yml ]; then
  read -rsp "Paste the bot account's Lichess token and press Enter (it won't show up as you paste) " TOKEN
  echo
  if [ -z "$TOKEN" ]; then
    echo "No token was given, so the config wasn't made. Run this again when you have it."
    exit 1
  fi
  sed -e "s|PASTE_YOUR_TOKEN_HERE|$TOKEN|" \
      -e 's|^  name: "markbot.exe".*|  # name: "markbot.exe"|' \
      -e 's|^  # name: "markbot.py"|  name: "markbot.py"|' \
      -e 's|^  # interpreter: |  interpreter: |' \
      deploy/lichess-bot-config.yml > lichess-bot/config.yml
  chmod 600 lichess-bot/config.yml   # only my user can read the token
fi
chmod +x engine/markbot.py           # lichess-bot won't start an engine file that isn't marked executable

# Quick test before going live, MarkBot should answer the first move of a game.
if ! printf 'uci\nisready\nposition startpos\ngo\nquit\n' | .venv/bin/python engine/markbot.py | grep -q bestmove; then
  echo "MarkBot didn't answer a test move, so the service wasn't installed. Check the models folder."
  exit 1
fi

# Fill my username and this folder into the service file and hand it to systemd.
echo "Installing the markbot service..."
sed -e "s|@USER@|$USER|g" -e "s|@REPO@|$REPO|g" deploy/markbot.service \
  | sudo tee /etc/systemd/system/markbot.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable -q markbot
sudo systemctl restart markbot

echo
echo "MarkBot is online and will come back by itself after a reboot or a crash."
echo "sudo systemctl status markbot     shows whether it's running"
echo "journalctl -u markbot -f          shows its live log, Ctrl+C to stop watching"
echo "sudo systemctl stop markbot       takes it offline until the next restart or boot"
