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
- [ ] Day 2: training data prep with recency and time-control weighting
- [ ] Day 3: fine-tuning script and first training run
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
python training/fetch_games.py GMMark12
```
This saves every game to `data/games.pgn`. It uses only the Python standard library.

## Credits
- [Maia-2](https://github.com/CSSLab/maia2) (Tang et al., NeurIPS 2024): the base human-move model
- [python-chess](https://github.com/niklasf/python-chess)
- [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)
