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
- [x] Day 4: opening book builder
- [x] Day 5: UCI engine and ONNX export
- [x] Day 6: held-out evaluation
- [x] Day 7: Windows setup scripts
- [x] Day 8: lichess-bot config and launcher
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
python training/build_book.py GMMark12    # opening book -> models/book.json
python training/export_onnx.py            # lightweight model -> models/mark_model.onnx
python training/evaluate.py GMMark12      # how well it matches me -> results/eval_results.json
```
On Windows we can do the quick start without having to manually input all of the above commands. `windows\setup.bat` makes a `.venv` and installs everything
once. `windows\retrain.bat` then runs all six steps above in
order and stops if one fails. So retraining on my newest games is very easy and just takes about 45 minutes of
waiting. It builds the bot's opening book from every game (`--holdout 0`), and the evaluation builds its own
book without the test games, so the results stay fair. A retrain overwrites `results/train_log.txt` and
`results/eval_results.json` with the new run.
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

`build_book.py` records what I actually play in the first 12 moves of my games, so the bot can open the way
I do instead of guessing. If I've reached a position in at least 3 games, the bot picks from the moves I
played there (anything I've played at least twice), about as often as I played each one, with recent games
counting more. Once a game leaves my usual lines, the neural net takes over.

By default the book skips my 150 most recent games, so it doesn't look at the test set. For the version the
bot actually plays with, I build it from every game with `--holdout 0`.

## Playing against it
`engine/markbot.py` is the bot itself. Its in UCI, the language chess programs use to talk to engines,
so it works in any chess GUI and with lichess-bot. When it's the bot's turn, it checks my opening book first and
if the position isn't there, it asks the model which moves I'd likely play at an 1800 level and picks one based on those chances. It doesn't calculate ahead like Stockfish. It just plays the move I would which includes the same blunders I make.

The engine runs on ONNX instead of PyTorch, so it only needs three small packages and picks a move in a
few milliseconds. It needs `mark_model.onnx`, `moves.json` and `book.json` in the `models/` folder.

Some chess apps, like En Croissant on Windows, only load `.exe` engines, so `engine/build_exe.bat` packs
the bot into a single `engine/markbot.exe` with PyInstaller. Double-click the script, then add `engine/markbot.exe` as a UCI engine in the chess app. It still reads the model files from `models/`, so keep the `.exe` in the `engine/` folder.

Apps that can run a Python script directly, like Cute Chess, can use it without building anything:
```
pip install -r engine/requirements.txt
python engine/markbot.py
```
In the app's engine settings you can change its rating, turn the opening book off, or turn down the
randomness so it always plays my single most likely move.

## On Lichess
You can play against the bot on Lichess by searching for MarkBot12 and clicking Challenge. It only plays real
people, one game at a time, in bullet, blitz, rapid or classical, and the games can be casual or rated. It never
resigns or offers a draw since it has no idea when a game is lost, so every game gets played out to the end.
If you want to run it yourself, `windows\lichess_bot.bat` downloads
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot), sets up its config from
`deploy/lichess-bot-config.yml` and puts the bot online. You'll need a brand new Lichess account for the bot
and a token for it.

## How well it plays like me
`training/evaluate.py` tests the bot on my 150 most recent games, which it never trained on. For every
position where it was my turn, it checks whether the bot's top choice is the move I actually played, and
compares that with plain Maia-2, the model I started from. The opening book it uses for this is built
without those games as well.

| | All moves | First 12 moves | After move 12 |
|---|---|---|---|
| Plain Maia-2 | 49.4% | 43.4% | 53.1% |
| My fine-tuned model | 54.8% | 57.5% | 53.1% |
| MarkBot (book + model) | 55.3% | 59.0% | 53.1% |

That's 4,615 positions from games played between February 2024 and October 2026, and "first 12 moves" is
the same stretch my opening book covers. All of the gain is in the opening, where I tend to make the same moves so the
bot goes from guessing my move 43% of the time to 59%. After move 12 it does no better than plain Maia-2, so
my middlegame and endgame moves look a lot like any other ~1800 player's, at least as far as 2,000 games can
show. Nobody plays the same move in the same spot every time, so 100% isn't possible.

## Credits
- [Maia-2](https://github.com/CSSLab/maia2) (Tang et al., NeurIPS 2024): the base human-move model
- [python-chess](https://github.com/niklasf/python-chess)
- [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)
