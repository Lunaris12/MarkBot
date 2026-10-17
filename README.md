# MarkBot

A chess bot that plays like me (chess.com: **GMMark12**, ~1800 rapid), not like a perfect engine.

The goal is a bot that friends can play on Lichess and that feels like playing me: my openings, my habits, and my mistakes.

## Plan
1. **Fetch games:** download all my chess.com games.
2. **Opening book:** record what I actually play in the opening, weighted toward recent games.
3. **Fine-tune:** start from [Maia-2](https://github.com/CSSLab/maia2), a model trained to predict the moves human players make at each rating,
   and train it further on my own moves.
4. **Play:** wrap it as a UCI engine that runs at an 1800 rating, usable in any chess GUI or as a Lichess bot.

To check that it actually plays like me, I'll hold out my most recent games from training and measure how often the bot
picks the move I really played.

## Progress
- [x] Day 1: project setup, game fetcher (`training/fetch_games.py`)
- [x] Day 2: training data prep with recency and time-control weighting
- [x] Day 3: fine-tuning script and first training run
- [ ] Day 4: opening book builder
- [ ] Day 5: UCI engine and ONNX export
- [ ] Day 6: held-out evaluation
- [ ] Day 7: Windows setup scripts
- [ ] Day 8: lichess-bot config and deploy docs
- [ ] Day 9: server setup script and systemd service
- [ ] Day 10: bot live on Lichess around the clock
- [ ] Day 11: experiment: rapid-only vs. all games
- [ ] Day 12: set the opponent rating from the real Lichess opponent
- [ ] Day 13: automatic monthly retraining on new games
- [ ] Day 14: accuracy-by-move-number chart and write-up

## Quick start (so far)
```
pip install -r training/requirements.txt
pip install --no-deps maia2==0.10.0
python training/fetch_games.py GMMark12   # every game -> data/games.pgn
python training/prep_data.py GMMark12     # training examples -> data/dataset.pt
python training/finetune.py               # fine-tuned model -> models/mark_model.pt
```
`prep_data.py` turns each position where I was to move into a training example.
Recent games and rapid games count more, and moves made with under 10 seconds left are dropped.
My 150 most recent games are kept aside as a test set and are never trained on.

`finetune.py` takes Maia-2's rapid model, which already plays like a typical human at each rating,
and keeps training it on my games so it drifts toward playing like me. Moves from recent and rapid games
pull harder than moves from old blitz games.

To keep training honest, I hold back the 100 games right before the test set. Every 250 steps the script
checks how often the model would have picked the move I actually played in those games, and it only saves
the model when that beats its previous best. The test games stay untouched until evaluation, so they're a
fair final check.

A full run (two passes over the data) takes about 40 minutes on a regular CPU, and it'll use a GPU
automatically if PyTorch finds one. The log from my first run is in `results/train_log.txt`.

## Credits
- [Maia-2](https://github.com/CSSLab/maia2) (Tang et al., NeurIPS 2024): the base human-move model
- [python-chess](https://github.com/niklasf/python-chess)
- [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)
