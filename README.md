# MarkBot

A chess bot that plays like me (GMMark12 on chess.com, around 1800 rapid) instead of like a perfect engine.
The idea is that my friends can play it on Lichess and it feels like playing me, with my openings, my habits
and my mistakes.

## Plan
I download all my chess.com games, build an opening book from what I actually play, and then take
[Maia-2](https://github.com/CSSLab/maia2), a model that already predicts the moves real people make at each
rating, and train it further on my own moves. That gets wrapped up as a UCI engine that plays at 1800 in any
chess app or on Lichess. To check it really plays like me, I keep my most recent games out of training and see
how often the bot picks the move I actually played.

## Progress
- [x] Day 1, project setup and game fetcher
- [x] Day 2, training data prep with recency and time control weighting
- [x] Day 3, fine-tuning script and first training run
- [x] Day 4, opening book builder
- [x] Day 5, UCI engine and ONNX export
- [x] Day 6, held-out evaluation
- [x] Day 7, Windows setup scripts
- [x] Day 8, lichess-bot config and launcher
- [x] Day 9, server setup script and systemd service
- [x] Day 10, bot live on Lichess around the clock
- [ ] Day 11, experiment with rapid-only vs all games
- [ ] Day 12, set the opponent rating from the real Lichess opponent
- [ ] Day 13, automatic monthly retraining on new games
- [ ] Day 14, accuracy by move number chart and write-up

## Quick start
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
On Windows I don't have to do all of that. `windows\setup.bat` makes a `.venv` and installs everything once,
and `windows\retrain.bat` runs all six steps in order and stops if one fails, so retraining on my newest games
is just about 45 minutes of waiting. It builds the bot's opening book from every game (`--holdout 0`) while the
evaluation builds its own book without the test games, so the results stay fair. Each retrain overwrites
`results/train_log.txt` and `results/eval_results.json` with the new run.

## How it works
`prep_data.py` turns every position where I was to move into a training example. Recent games and rapid games
count more, and moves I made with under 10 seconds left get dropped since those are mostly panic moves. My 150
most recent games are kept aside as a test set and never get trained on.

`finetune.py` takes Maia-2's rapid model, which already plays like a typical human at each rating, and keeps
training it on my games so it drifts toward playing like me. Moves from recent and rapid games pull harder than
moves from old blitz games. To keep training honest I also hold back the 100 games right before the test set.
Every 250 steps the script checks how often the model would have picked my real move in those games, and it
only saves the model when that beats its previous best. A full run (two passes over the data) takes about 40
minutes on a regular CPU, and it uses a GPU automatically if PyTorch finds one. The log from my first run is
in `results/train_log.txt`.

`build_book.py` records what I actually play in the first 12 moves of my games, so the bot opens the way I do
instead of guessing. If I've reached a position in at least 3 games, the bot picks from the moves I played
there at least twice, about as often as I played each one, with recent games counting more. Once a game leaves
my usual lines the neural net takes over. By default the book skips my 150 most recent games so it doesn't
peek at the test set, and the version the bot actually plays with is built from every game with `--holdout 0`.

## Playing against it
`engine/markbot.py` is the bot itself. It's in UCI, the language chess programs use to talk to engines, so it
works in any chess app and with lichess-bot. On its turn it checks my opening book first, and if the position
isn't there it asks the model which moves I'd likely play at 1800 and picks one based on those chances. It
doesn't calculate ahead like Stockfish. It just plays the move I would, blunders included.

The engine runs on ONNX instead of PyTorch, so it only needs three small packages and picks a move in a few
milliseconds. It needs `mark_model.onnx`, `moves.json` and `book.json` in the `models/` folder.

Some chess apps, like En Croissant on Windows, only load `.exe` engines, so `engine/build_exe.bat` packs the bot
into a single `engine/markbot.exe` with PyInstaller. Double-click the script and add `engine/markbot.exe` as a
UCI engine in the app. It still reads the model files from `models/`, so keep the `.exe` in the `engine/` folder.
Apps that can run a Python script directly, like Cute Chess, can use it without building anything.
```
pip install -r engine/requirements.txt
python engine/markbot.py
```
In the app's engine settings you can change its rating, turn the opening book off, or turn down the randomness
so it always plays my single most likely move.

## On Lichess
MarkBot is online on Lichess all the time as [MarkBot12](https://lichess.org/@/MarkBot12), so you can play it
whenever by going to its profile and clicking Challenge. It only plays real people, one game at a time, in
bullet, blitz, rapid or classical, and the games can be casual or rated. It never resigns or offers a draw since
it has no idea when a game is lost, so every game gets played out to the end. If you want to run it yourself,
`windows\lichess_bot.bat` downloads [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot), sets up its
config from `deploy/lichess-bot-config.yml` and puts the bot online. You'll need a brand new Lichess account
for the bot and a token for it.

## Running it on a server
My PC can't stay on all day, so the bot runs on a small free Linux server from Google instead. My model files
aren't on GitHub, so they get copied over from my PC by hand. After that `deploy/setup_server.sh` installs
everything, asks for the bot's token once and sets MarkBot up as a systemd service, which starts the bot
whenever the server boots and restarts it if it crashes.
```
git clone https://github.com/Lunaris12/MarkBot.git && mkdir -p MarkBot/models     # on the server
scp models/mark_model.onnx models/moves.json models/book.json USER@SERVER_IP:MarkBot/models/   # on my PC
bash MarkBot/deploy/setup_server.sh                                                # on the server
```
`sudo systemctl status markbot` shows if it's running and `journalctl -u markbot -f` shows its live log. When
I change the code, I push it to GitHub and run `bash MarkBot/deploy/update_server.sh` on the server, which pulls
the new version and restarts the bot. If I retrain, I copy the new model files over the same way and run the
same script.

## How well it plays like me
`training/evaluate.py` tests the bot on my 150 most recent games, which it never trained on. For every position
where it was my turn, it checks whether the bot's top choice is the move I actually played, and compares that
with plain Maia-2, the model I started from. The opening book it uses for this is built without those games too.

| | All moves | First 12 moves | After move 12 |
|---|---|---|---|
| Plain Maia-2 | 49.4% | 43.4% | 53.1% |
| My fine-tuned model | 54.8% | 57.5% | 53.1% |
| MarkBot (book + model) | 55.3% | 59.0% | 53.1% |

That's 4,615 positions from games played between February 2024 and October 2026, and the first 12 moves are the
same stretch my opening book covers. All of the gain is in the opening, where I tend to play the same moves, so
the bot goes from guessing my move 43% of the time to 59%. After move 12 it does no better than plain Maia-2, so
my middlegame and endgame moves look a lot like any other 1800 player's, at least as far as 2,000 games can
show. Nobody plays the same move in the same spot every time, so 100% isn't possible.

## Credits
MarkBot is built on [Maia-2](https://github.com/CSSLab/maia2) (Tang et al., NeurIPS 2024), the human move
model I fine-tuned, and uses [python-chess](https://github.com/niklasf/python-chess) for the chess logic and
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) to connect it to Lichess.
