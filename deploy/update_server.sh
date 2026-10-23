#!/usr/bin/env bash
# This updates MarkBot on the server after I push new code to GitHub.
# It pulls the latest version of the repo, reinstalls the engine packages in case they changed and
# restarts the service so the bot picks everything up. New model files still get copied over from my
# PC by hand since they aren't on GitHub, and running this afterwards puts them live too.
# Restarting cuts off any game the bot is in the middle of, so I run it when it isn't playing.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Getting the latest code..."
git pull --ff-only
.venv/bin/pip install -q -r engine/requirements.txt
chmod +x engine/markbot.py

echo "Restarting MarkBot..."
sudo systemctl restart markbot
sleep 5   # give it a few seconds to start up or crash
if systemctl is-active --quiet markbot; then
  echo "MarkBot is back online."
else
  echo "MarkBot didn't come back up. journalctl -u markbot -n 50 shows what went wrong."
  exit 1
fi
