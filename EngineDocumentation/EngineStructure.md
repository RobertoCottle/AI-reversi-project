This is a substantial architecture — let me walk through it as interconnected layers, starting with the foundation and building upward. I'll use diagrams to anchor the key structural relationships.

---

## Layer 1: The game engine — the foundation everything runs on

The very first thing you build, before any neural network touches code, is a correct and fast `ReversiEnv`. This is your source of truth. It encapsulates the 8×8 bitboard (representing black pieces, white pieces, and legal moves as 64-bit integers rather than nested Python lists — this is critical for speed), the `step(action)` method that applies a move and flips discs, the `legal_moves()` generator, and terminal detection. Every single component in the system will call into this, so you want it tested to exhaustion against known game transcripts before writing anything else.

The bitboard representation deserves particular emphasis. A Python list-of-lists board state is a training bottleneck. A pair of 64-bit integers can compute all legal moves in microseconds using bitwise shift masks, and can be hashed for deduplication in a single operation. This matters enormously when you're generating millions of positions per hour.

---

## Layer 2: Phase one training — the Q/TD bootstrapper

Rather than starting AlphaZero cold, you pre-train a lightweight value estimator using TD learning on n-tuple networks. This component is entirely separate from the deep network — it runs on CPU, trains in hours rather than days, and produces a warm initialization for the value function.

The `NTupleNetwork` is a lookup-table-based function approximator. You define a set of board patterns — say, all horizontal and diagonal 4-tuples, giving you roughly 288 weights — and for each game state, you activate the relevant tuples and sum their stored values. TD learning updates these weights after each self-play game: starting from terminal outcome +1 or -1, you propagate backwards through the move history, nudging each intermediate state estimate toward its successor. The learning rule is dead simple and converges reliably because there's no neural backpropagation involved.

After approximately 50,000 self-play games (achievable in under two hours on a single CPU thread), this network plays at intermediate strength — reliably avoiding immediate losing moves, respecting corner priority, and maintaining basic mobility. Its value estimates across different board positions are now meaningful enough to serve as initialization targets.

The Q-learning variant of this phase adds experience replay. Instead of purely sequential TD updates, you store `(state, action, reward, next_state)` tuples in a circular buffer and sample random minibatches. This breaks the temporal correlation that makes pure TD oscillate and produces smoother convergence on complex mid-game evaluations. However, for the warmup purpose, pure TD on n-tuples is usually sufficient.

---

## Layer 3: The deep network — ResNet policy-value tower

Now you build the actual neural network that AlphaZero will train. The architecture is a single `PolicyValueNet` that takes a board state and outputs two things simultaneously: a policy vector (probability distribution over all 65 possible moves — 64 squares plus pass) and a scalar value (expected game outcome from the current player's perspective, in [-1, +1]).

The input encoding deserves careful thought. You represent the board as a stack of binary planes: plane 0 is current player's pieces, plane 1 is opponent's pieces, plane 2 is legal moves mask, and optionally planes 3-8 encode the last three moves for each player (giving temporal context). This gives an 8×8×6 tensor that a convolutional architecture can process naturally.

The body of the network is a stack of residual blocks — typically 6-10 for Reversi given the small board, each block consisting of two 3×3 convolutional layers with batch normalization and ReLU activations around a skip connection. The residual architecture prevents gradient vanishing during deep backpropagation and allows the network to learn identity mappings (essentially "this board feature is already fine, pass it through unchanged"), which is valuable when many mid-game positions are subtle variations of similar strategic configurations.

The policy head branches off the residual tower with a single 1×1 conv followed by a fully-connected layer to 65 outputs and a softmax. The value head does the same but terminates in a single tanh output. Both heads are trained jointly from the same gradient, which means improvements in value estimation also benefit policy learning and vice versa — they share all the spatial feature extraction work.

You initialize the weights of this network not randomly, but by supervised pre-training: you run the TD n-tuple agent for its 50,000 games, record the final n-tuple value estimate for every position seen, and train the ResNet to regress onto those values. This gives the network a reasonable starting point for value estimates across all game phases, resolving the cold-start problem that plagues standard AlphaZero.

---

## Layer 4: MCTS — the search engine

The `MonteCarloTreeSearch` component operates at inference time (and during self-play data generation, but not during gradient updates). It maintains a tree of `MCTSNode` objects, each storing visit count `N(s,a)`, total action value `W(s,a)`, mean action value `Q(s,a)`, and the policy prior `P(s,a)` predicted by the neural network.

The selection step traverses from the root by repeatedly choosing the action that maximizes the PUCT score: `Q(s,a) + c_puct × P(s,a) × √(Σ_b N(s,b)) / (1 + N(s,a))`. The exploration constant `c_puct` (typically around 1.5–2.0 for Reversi) controls how much the prior probability pulls exploration away from the currently highest-valued actions. When a leaf node is reached, the neural network evaluates it, returning both the policy vector (which seeds the children's priors) and the value estimate (which is backpropagated up the tree, updating `W` and `N` along the path).

After 400–800 simulations (the budget you allocate per move during self-play), the action distribution is taken as `π(a|s) = N(s,a)^(1/τ) / Σ_b N(s,b)^(1/τ)` where temperature τ controls how sharply the distribution concentrates. During the opening phase of a game (first 15 moves) you use τ=1 for exploration; in the endgame you drop to τ→0, making the agent near-deterministic on the highest-visited action.

This is the component where the Go-Exploit improvement integrates. Instead of always starting self-play games from the standard opening position, you maintain an `ArchiveBuffer` of interesting board positions — states where the MCTS value estimate was uncertain (high variance across visits), states at game-phase boundaries, or states sampled from human tournament databases. You periodically start self-play trajectories from these archived positions, generating shorter trajectories that focus network training on the regions of the game tree where value estimates are most inaccurate.

---

## Layer 5: The self-play pipeline — data generation

The `SelfPlayWorker` is the engine room of training. It runs as an asynchronous process (you typically run 4-8 workers in parallel, each playing complete games) that continuously generates training examples and pushes them into the `ReplayBuffer`.

Each game proceeds as follows: from the current position, run MCTS for the allocated simulation budget, sample an action from the resulting π distribution, apply it to `ReversiEnv`, and record the tuple `(board_state, π, player_to_move)`. Repeat until terminal. At game end, annotate every recorded tuple with the final outcome z ∈ {-1, 0, +1} from the perspective of the player who was to move in that position. This gives you a full game's worth of `(board_state, π, z)` triples to push into the replay buffer.

The replay buffer itself is a fixed-size circular deque holding the most recent N complete games (typically N=500,000 positions). It supports efficient uniform random sampling. You deduplicate positions using their bitboard hash before insertion, which both removes redundancy and slightly corrects the training distribution skew caused by common opening sequences appearing in nearly every game.

The endgame-first curriculum integrates here. During the early training phase (first 200 iterations), the workers occasionally start games at randomly sampled positions from the final 20 moves of previously played games. This ensures the network receives dense signal about terminal evaluations before attempting to generalize to openings — the intuition being that a network that understands endgames can correctly assign credit to the mid-game positions that led there, but the reverse is not true.

---

## Layer 6: The training loop — gradient updates

The `TrainingEngine` runs on the GPU and consumes batches from the `ReplayBuffer`. For each batch of 512 positions, it computes:

The policy loss as cross-entropy between the network's predicted policy `p` and the MCTS-improved policy `π`: `L_policy = -Σ π(a) log p(a)`. The value loss as mean-squared error between predicted value `v` and actual outcome `z`: `L_value = (v - z)²`. An L2 regularization term on network weights to prevent overfitting. The total loss is `L = L_policy + L_value + λ × L_regularization`.

You optimize with Adam (learning rate starting at 2×10⁻³, decaying by 0.1× at iteration milestones 200 and 400). Gradient clipping at norm 1.0 prevents occasional explosive updates during early training when the value estimates are still unreliable.

The MuZero Reanalyze improvement plugs in here. Every few training steps, instead of drawing a fresh minibatch from the replay buffer, you take an existing batch of positions and re-run MCTS on them using the current (improved) network weights to generate fresh π targets. This means old game positions continuously receive updated, higher-quality training signals as the network improves, rather than being stuck with the policy estimates from when they were first played. The computational overhead is moderate — roughly 30% more inference cost — but the sample efficiency gain is substantial.

---

## Layer 7: The evaluation harness — testing

You maintain a separate `EvaluationEngine` that runs every 50 training iterations and produces the metrics that tell you whether training is actually progressing.

The primary benchmark is win rate against a versioned set of fixed opponents. You maintain a ladder: `RandomAgent` (wins ≥ 99% immediately), `GreedyFlipAgent` (always maximizes immediate disc count), `AlphaBetaAgent` configured at depths 1, 3, and 5, `EdaxAgent` at levels 1 through 7 (each level roughly doubling search depth), and `PreviousSelf` (the network from 50 iterations ago). Each evaluation match plays 40 games (20 as black, 20 as white) to control for the known first-player advantage in Reversi. A win rate below 55% against the previous self version is a signal that training has stalled and hyperparameters need adjustment.

Secondary metrics instrument the internals. Value prediction error measures how far the network's raw value estimate (before MCTS search) diverges from actual game outcomes — this should decrease monotonically. Policy entropy tracks how confident the policy head is becoming; it should decrease over training but not collapse to near-zero too early, which would indicate the network is overfitting to a narrow set of openings. MCTS visit distribution analysis checks whether the agent is allocating simulation budget proportionally to move quality or getting stuck in local minima.

You also maintain a `BoardStateAnalyzer` that periodically samples 1,000 positions from the replay buffer and plots value estimate distributions broken down by game phase (opening, midgame, endgame). A well-trained network should show tight, confident value estimates in the endgame and progressively wider uncertainty distributions as you move backward toward the opening — if the pattern is inverted, the training curriculum needs adjustment.

---

## Layer 8: The competition agent — inference

For actual play against other AIs, the `CompetitionAgent` wraps everything together and exposes a single `get_action(board, time_budget_ms)` interface.

Given a board state, it first queries the `OpeningBook` — a precomputed database of strong opening sequences extracted from human tournament games and known optimal lines. If the current position matches a book entry, the move is returned immediately with no neural network involvement.

Out of book, it runs MCTS with a simulation count calibrated to the available time budget. At 5 seconds per move (typical for online competition), you run approximately 1,600 simulations, using the neural network for leaf evaluation. The temperature is set to τ→0 (play the most-visited action deterministically). Symmetry augmentation is applied: the board is evaluated in all 8 symmetries (4 rotations × 2 reflections), the network is queried for each, and the policy/value outputs are averaged before seeding the MCTS tree. This is essentially free ensemble inference that consistently improves move quality by 50–100 Elo points.

For very short time budgets (under 1 second), you fall back to `QuickEvalMode` which skips MCTS entirely and plays directly according to the raw network policy output, further sharpened by a single-pass alpha-beta search at depth 3 using the network's value head as the evaluation function. This degrades gracefully rather than timing out.

The agent also runs an `OpponentModeler` that tracks the opponent's move history across the game, comparing their actual moves to what the network would predict. Systematic deviations from expected play reveal opponent tendencies (e.g., always prioritizing mobility, consistently avoiding X-squares) which can inform move selection in ambiguous positions.

---

Here's the full structural view of how these layers relate at runtime:---

## The training data lifecycle in detail

It's worth being precise about what a "training example" actually is at each stage, because the data format changes across the phases.

During the TD bootstrapping phase, each training example is simply a `(bitboard_state, td_value_target)` pair — a float in [-1, +1] produced by the retrospective n-tuple update. No policy information is captured here; you're only building a value map.

During the supervised pretraining of the ResNet, you replay all those bootstrapped positions through the new network and compute MSE loss between predicted value and the n-tuple estimate. You run this for roughly 20 epochs until the residual blocks have learned to approximate the n-tuple's value function.

Once the AlphaZero self-play loop starts, the training format shifts. Each example becomes a triple `(board_planes, mcts_policy_π, game_outcome_z)`. The board planes are the 8×8×6 tensor described earlier. The MCTS policy π is the normalized visit count distribution over all 65 possible actions, shaped by the temperature schedule. The game outcome z is the final win/loss/draw from the perspective of the player to move in that position — this is only known at game end and is applied retrospectively to all positions in the game trajectory.

The training loop sees batches of 512 such triples, randomly sampled from the replay buffer. The two loss terms — cross-entropy against π and MSE against z — are summed and backpropagated together. An important subtlety: during MuZero Reanalyze, you re-run MCTS on old board states using the current network weights, producing a fresh π. This fresh π replaces the originally stored π for those positions in the current minibatch, so the policy gradient always reflects the network's current best understanding rather than its understanding at the time the game was originally played.

---

## Testing vs performance: a clear division

Testing is the process of measuring whether training converged correctly. You run it controlled and offline — fixed opponents, fixed opening positions, logging everything. Performance is live play under competitive conditions with time pressure and adaptive opponents. The two modes make different demands on the system.

In testing mode, you care about reproducibility (fixed random seeds, deterministic MCTS), full logging (every MCTS tree dump, every policy distribution, every value estimate), and breadth of opponents. The `EvaluationEngine` should be able to compare two model checkpoints head-to-head, run statistical significance tests on win rate differences (you need at least 200 games to establish a 5% win rate margin at p<0.05), and produce the policy entropy and value accuracy curves that let you diagnose training pathologies.

In performance mode, the constraints invert. Reproducibility is irrelevant; speed and robustness under time pressure are everything. The `CompetitionAgent` strips out all logging, uses the opening book aggressively to save time budget for critical mid-game decisions, and runs the symmetry ensemble only on moves where MCTS visits are concentrated (high confidence), skipping it when time is tight. The QuickEval fallback ensures the agent always moves legally even if the time budget has nearly expired.

The transition from testing to competition happens through a single configuration switch — the underlying `PolicyValueNet` and `MonteCarloTreeSearch` are shared; only the wrapper layer changes. This is the practical argument for keeping the architecture cleanly separated across layers: you never want to accidentally run a logging-heavy evaluation mode in a time-critical competition match.

//-------------- project structure ----------------------------------//

Great questions — let me answer them in full. The short answers are: no database needed, no external services required, and the whole project runs entirely from VS Code and Python. Here's what it all looks like.

---

## Project structure

The layout follows a clean separation between the game logic, model code, training pipeline, evaluation harness, and saved artifacts. Everything lives under one root folder.

```
reversi_ai/
│
├── reversi_ai/                  ← main Python package
│   ├── __init__.py
│   │
│   ├── env/                     ← Layer 1: game engine
│   │   ├── __init__.py
│   │   ├── board.py             ← ReversiEnv, bitboard logic
│   │   ├── moves.py             ← legal move generation, flip masks
│   │   └── constants.py         ← board size, directions, masks
│   │
│   ├── bootstrap/               ← Layer 2: TD / n-tuple pre-trainer
│   │   ├── __init__.py
│   │   ├── ntuple_network.py    ← NTupleNetwork, weight lookup tables
│   │   ├── td_trainer.py        ← TDTrainer, self-play, weight updates
│   │   └── replay_buffer.py     ← simple circular buffer for TD phase
│   │
│   ├── model/                   ← Layer 3: ResNet policy-value network
│   │   ├── __init__.py
│   │   ├── network.py           ← PolicyValueNet (ResNet body + heads)
│   │   ├── encoder.py           ← board → 8×8×6 tensor conversion
│   │   └── pretrain.py          ← supervised warmup from n-tuple values
│   │
│   ├── search/                  ← Layer 4: MCTS
│   │   ├── __init__.py
│   │   ├── mcts.py              ← MonteCarloTreeSearch, MCTSNode
│   │   ├── puct.py              ← PUCT selection formula
│   │   └── archive.py           ← ArchiveBuffer for Go-Exploit starts
│   │
│   ├── training/                ← Layers 5–6: self-play + gradient updates
│   │   ├── __init__.py
│   │   ├── self_play.py         ← SelfPlayWorker, game loop
│   │   ├── replay_store.py      ← ReplayBuffer (large, on-disk backed)
│   │   ├── trainer.py           ← TrainingEngine, loss functions, Adam
│   │   └── reanalyze.py         ← MuZero-style reanalysis pass
│   │
│   ├── evaluation/              ← Layer 7: testing harness
│   │   ├── __init__.py
│   │   ├── evaluator.py         ← EvaluationEngine, match runner
│   │   ├── opponents.py         ← RandomAgent, GreedyAgent, AlphaBetaAgent
│   │   ├── edax_wrapper.py      ← subprocess wrapper around Edax binary
│   │   └── metrics.py           ← win rate, value error, policy entropy
│   │
│   ├── agent/                   ← Layer 8: competition agent
│   │   ├── __init__.py
│   │   ├── competition.py       ← CompetitionAgent, time-aware interface
│   │   ├── opening_book.py      ← OpeningBook, hash lookup
│   │   └── symmetry.py          ← 8-fold symmetry augmentation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py            ← dataclass-based config loading
│       ├── checkpoint.py        ← save/load model weights + training state
│       └── logging.py           ← structured training log writer
│
├── scripts/                     ← entry points you actually run
│   ├── run_bootstrap.py         ← Phase 1: run TD pre-training
│   ├── run_pretrain.py          ← Phase 2: supervised warmup of ResNet
│   ├── run_training.py          ← Phase 3: full AlphaZero loop
│   ├── run_evaluation.py        ← evaluate a checkpoint against opponents
│   └── run_play.py              ← human vs agent, or agent vs agent
│
├── configs/                     ← YAML / TOML config files
│   ├── default.toml             ← baseline hyperparameters
│   ├── fast_dev.toml            ← small model, quick iteration
│   └── competition.toml         ← high-sim-count competition settings
│
├── data/                        ← all saved artifacts (no database needed)
│   ├── checkpoints/             ← model weight files (.pt)
│   │   ├── ntuple_weights.pkl   ← serialized n-tuple lookup table
│   │   ├── iter_0050.pt         ← ResNet checkpoint at iteration 50
│   │   ├── iter_0100.pt
│   │   └── best.pt              ← symlink to current best checkpoint
│   │
│   ├── replay_buffer/           ← on-disk replay buffer
│   │   └── buffer.npz           ← compressed NumPy arrays (board, pi, z)
│   │
│   ├── opening_book/
│   │   └── book.pkl             ← dict mapping position hash → best move
│   │
│   ├── training_logs/           ← CSV + JSON logs per run
│   │   ├── run_20250401/
│   │   │   ├── training.csv     ← loss, lr, iteration per step
│   │   │   └── eval.csv         ← win rates per evaluation checkpoint
│   │   └── run_20250402/
│   │
│   └── games/                   ← optional: saved SGF-style game records
│       └── self_play_iter050.pkl
│
├── tests/                       ← unit and integration tests
│   ├── test_board.py            ← board correctness, known game transcripts
│   ├── test_mcts.py             ← MCTS visit counts, value propagation
│   ├── test_network.py          ← network output shapes, valid probabilities
│   └── test_training.py         ← one iteration runs without error
│
├── notebooks/                   ← optional: analysis and visualization
│   ├── training_curves.ipynb
│   └── position_analysis.ipynb
│
├── requirements.txt
├── pyproject.toml               ← package definition, tool config
└── README.md
```

Here's a visual map of how the packages relate at runtime:---

## Do you need anything outside VS Code and Python?

Almost nothing. The entire project runs with Python 3.10+ and a standard `pip install`. Here is every dependency and exactly why it's needed:

`torch` and `torchvision` are the only heavy dependencies. PyTorch handles all the neural network forward/backward passes, GPU acceleration (if you have a CUDA-capable card), and tensor operations. You don't need TensorFlow, JAX, or anything else alongside it.

`numpy` is used everywhere for bitboard operations, replay buffer storage as compressed arrays, and any numerical work that doesn't need automatic differentiation. It's implicit in PyTorch but you'll use it directly in the game engine and buffer serialization.

`multiprocessing` from the Python standard library handles the parallel self-play workers. No external job queue needed — Python's built-in process pool is sufficient because the workers are embarrassingly parallel and communicate only by pushing completed game records to a shared queue.

`toml` or `tomllib` (built into Python 3.11+) handles configuration file parsing. You define all hyperparameters in `.toml` files and load them as typed dataclasses.

`pytest` is your test runner. No extras needed.

`matplotlib` is optional but useful for plotting training curves from the CSV logs, either in standalone scripts or in Jupyter notebooks. If you want interactive plots during training, `tensorboard` is a light addition — you write scalars with a few lines and open the dashboard in a browser tab.

The one optional external binary is `Edax`, the strongest open-source Reversi engine, which you wrap via `subprocess` in `edax_wrapper.py`. It's a standalone compiled executable (~500KB), not a Python package. You only need it if you want to benchmark against Edax levels during evaluation. Everything else in the evaluation harness — `RandomAgent`, `GreedyAgent`, `AlphaBetaAgent` — is pure Python.

Your `requirements.txt` therefore looks like this:

```
torch>=2.2.0
numpy>=1.26.0
pytest>=8.0.0
matplotlib>=3.8.0        # optional, for plotting
tensorboard>=2.15.0      # optional, for live training dashboard
```

That's the complete dependency list. No Docker, no Redis, no message brokers, no cloud services.

---

## No database — files are the right choice here

A database would be the wrong tool for this project. Here is why, and what files replace each concern.

The replay buffer contains up to 500,000 training positions. Each position is a fixed-size NumPy array: an `(8, 8, 6)` float16 board tensor, a 65-element float32 policy vector, and a single float32 scalar value. The entire buffer serializes cleanly as a compressed `.npz` file via `numpy.savez_compressed`. Loading it back is a single line. Random sampling is done entirely in memory using `numpy.random.choice`. A database would add query overhead, a connection layer, and schema management for data that is fundamentally an array — exactly what NumPy was designed for.

The model checkpoints are PyTorch state dicts saved with `torch.save`. A checkpoint at iteration 100 is a single `iter_0100.pt` file containing all network weights, the optimizer state (so training can resume without any learning rate warmup), and metadata like the current iteration number and Elo estimates. `torch.load` restores it completely in under a second. You maintain a `best.pt` symlink (or simply a `best_iter.txt` file containing the iteration number) pointing to the current best checkpoint. There is no need to version this in a database.

The n-tuple network is a Python dictionary mapping tuple patterns to float weights. It serializes with `pickle` or `joblib` to a single `.pkl` file of a few megabytes. Load time is milliseconds.

The opening book is also a dictionary — board position hash (a 64-bit integer) mapping to a recommended move index. Same `.pkl` serialization.

Training logs are append-only CSV files, one row per training step or evaluation run. Because they're append-only and you never need to query them by arbitrary criteria during training, CSV is strictly better than a database here. You can open them in a spreadsheet, grep them from the terminal, or read them with `pandas` in a notebook. If you were building a web interface that needed to query training history by date range or model version, a database would make sense. But you're not — you're doing local research iteration.

The game records from self-play (if you choose to save them) are lists of move sequences serialized as `.pkl` files grouped by training iteration. Each file holds all games from one batch. This makes it easy to replay any game, inspect position statistics, or feed archived positions to the Go-Exploit archive buffer.

The complete `data/` directory after a full training run looks roughly like this:

```
data/
├── checkpoints/
│   ├── ntuple_weights.pkl          ~2 MB
│   ├── iter_0050.pt                ~45 MB
│   ├── iter_0100.pt                ~45 MB
│   ├── iter_0150.pt                ~45 MB
│   └── best.pt → iter_0150.pt
│
├── replay_buffer/
│   └── buffer.npz                  ~800 MB (500k positions compressed)
│
├── opening_book/
│   └── book.pkl                    ~5 MB
│
├── training_logs/
│   └── run_20250401/
│       ├── training.csv            ~1 MB (one row per batch)
│       └── eval.csv                ~10 KB (one row per evaluation)
│
└── games/
    ├── self_play_iter050.pkl       ~50 MB per iteration (optional)
    └── self_play_iter100.pkl
```

Total disk footprint for a complete run is roughly 2-4 GB, dominated by the replay buffer and checkpoint files. No database server process, no schema migrations, no connection strings. All of it survives a machine restart and resumes cleanly because every file contains everything needed to reconstruct state from that point.

The only time you would consider adding a database is if you wanted to run training across multiple machines (a distributed setup where workers on different machines push game records to a central store), or if you were building a web frontend to browse training history interactively. For a single-machine project running in VS Code, the file-based approach is simpler, faster to implement, and completely sufficient.

Yes — heuristics and game rules have very specific and important homes in this architecture. Let me address both questions precisely.

---

## Where game rules live

Game rules belong exclusively in `env/board.py` and `env/moves.py`. This is a hard boundary, not a convention. Every other component in the system — the MCTS, the training loop, the competition agent — must be completely ignorant of Reversi rules. They interact with the board only through the `ReversiEnv` interface: call `legal_moves()`, call `step(action)`, check `is_terminal()`. If any other module starts reasoning about disc flipping or adjacency directly, you have a coupling bug that will cause subtle errors when you later modify the game engine or try to port the architecture to a different game.

The `board.py` module owns the canonical state representation and the transition function. The `moves.py` module owns move legality — the bit manipulation that computes which squares are valid given the current board state. `constants.py` defines the eight directional shift masks (north, northeast, east, southeast, south, southwest, west, northwest) as 64-bit integers computed once at import time. These masks are the mathematical expression of the rule that a move is only legal if it flanks at least one opponent disc in some direction.

This is the only place in the entire project that "knows" the rules in a procedural sense.

---

## Where heuristics live — and there are three distinct kinds

Heuristics appear in three different places for three different purposes, and conflating them is a common design mistake.

The first kind is evaluation heuristics used by the classical opponents in `evaluation/opponents.py`. The `AlphaBetaAgent` needs a hand-crafted board scoring function to make alpha-beta search useful at shallow depths. This function lives in `evaluation/opponents.py` (or a small `evaluation/heuristics.py` module it imports) and encodes positional knowledge: corner squares score very high, X-squares score very low, stability counts score moderately, and mobility (number of legal moves available) scores positively. These weights are static, human-tuned constants. They exist only to give you a meaningful non-neural baseline during evaluation — they never touch the training pipeline.

The second kind is search heuristics baked into the MCTS itself, in `search/mcts.py`. Dirichlet noise added to root node priors during self-play is a heuristic for exploration. The temperature schedule — high entropy early in the game, near-deterministic late — is a heuristic governing how greedily the agent follows MCTS visit counts. The `c_puct` constant in the PUCT formula is a heuristic balancing exploration versus exploitation. These are not domain-knowledge heuristics about Reversi specifically; they are algorithmic parameters that govern search behavior in any game. They belong in `search/` as configuration values, not hardcoded constants.

The third kind is strategic heuristics encoded in the opening book and the competition agent. The `opening_book.py` is itself a large compiled heuristic — a lookup table of known strong moves in standard opening positions, derived from tournament databases and solved lines. The `symmetry.py` augmentation in the competition agent is a heuristic (averaging over 8 board symmetries improves value estimates without additional training). These live in `agent/` because they are inference-time policy enhancements, not training-time components.

Here is what the heuristics module in `evaluation/` looks like conceptually:

```python
# evaluation/heuristics.py

CORNER_POSITIONS = {0, 7, 56, 63}
XSQUARE_POSITIONS = {9, 14, 49, 54}  # diagonally adjacent to corners
CSQUARE_POSITIONS = {1, 6, 8, 15, 48, 55, 57, 62}  # edge-adjacent to corners

POSITIONAL_WEIGHTS = [
    100, -25,  10,   5,   5,  10, -25, 100,
    -25, -50,  -2,  -2,  -2,  -2, -50, -25,
     10,  -2,   5,   1,   1,   5,  -2,  10,
      5,  -2,   1,   2,   2,   1,  -2,   5,
      5,  -2,   1,   2,   2,   1,  -2,   5,
     10,  -2,   5,   1,   1,   5,  -2,  10,
    -25, -50,  -2,  -2,  -2,  -2, -50, -25,
    100, -25,  10,   5,   5,  10, -25, 100,
]

def evaluate_position(board, player):
    """Static board evaluator for AlphaBetaAgent."""
    score = 0
    score += positional_score(board, player)
    score += 5 * mobility_score(board, player)
    score += 10 * stability_score(board, player)
    return score
```

This is completely decoupled from the neural network. The ResNet has no access to these weights and does not use them during training — it learns its own internal evaluation from game outcomes. The classical heuristic function exists purely to give the `AlphaBetaAgent` something meaningful to optimize during alpha-beta search, so it can serve as a meaningful baseline during evaluation.

---

## What the competition agent depends on externally

This is the part of the architecture that actually touches the outside world, so it deserves careful accounting. Here is a complete breakdown:

The NBoard protocol interface is the single most important external dependency. Most online Reversi competition platforms (Reversiplorer, Cassio, BrowserOthello tournaments, and local tournament setups) use the NBoard 2.0 protocol — a simple text-based stdin/stdout protocol where the engine receives commands like `set game <moves>`, `move` (telling the agent to compute and output its move), `hint <n>` (request N best moves with scores), and `quit`. Your `CompetitionAgent` needs a thin `nboard_protocol.py` adapter in the `agent/` package that reads from stdin, parses these commands, calls into the MCTS engine, and writes moves to stdout in the expected format. This is a pure Python implementation — no external library needed, just the protocol spec, which is publicly documented.

The Edax binary is the second external dependency, but only during evaluation, not during competition. When testing in `run_evaluation.py`, your `edax_wrapper.py` launches Edax as a subprocess and communicates with it via NBoard protocol or a simpler move-exchange pipe. You do not need Edax compiled into your code — you call it as an external process. On Linux and macOS this is a straightforward subprocess call; on Windows you may need the Windows build of Edax, but it exists. The key detail is that Edax only belongs in the `evaluation/` package — the competition agent should never depend on it at runtime.

A CUDA-capable GPU is a hardware dependency rather than a software one, but it needs to be acknowledged. During competition, the agent's neural network forward pass (used at every MCTS leaf node evaluation) needs to complete within a few hundred microseconds to allow 1600 simulations in a 5-second time budget. On a modern GPU (even a consumer RTX 3060 or better), batched inference is fast enough. On CPU alone, you can run maybe 200-400 simulations in 5 seconds depending on the network size, which is still functional but weaker. The `CompetitionAgent` should check `torch.cuda.is_available()` at initialization and log a warning if running CPU-only — not crash, but degrade gracefully by reducing the simulation count budget.

PyTorch itself needs to be installed with the right CUDA version to match your driver. This is the one installation step that is slightly non-trivial. You install it with a platform-specific wheel: `pip install torch --index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1, for example. The `requirements.txt` should document this with a comment rather than trying to encode the CUDA version in the dependency string.

For online play in real competitions, you need a process management wrapper. If you are running the agent in a long tournament where it plays dozens of games sequentially, you want something that restarts the agent process if it crashes, logs all moves and time usage, and ensures the agent doesn't retain state between games. On Linux this is a simple shell script or `supervisord`. On Windows it is a batch file or a Task Scheduler entry. This is not a Python package dependency — it is an operational concern — but it belongs in a `scripts/run_competition.sh` (or `.bat`) file at the project root.

The full dependency picture for the competition agent specifically therefore looks like this:

```
CompetitionAgent runtime deps
├── torch                ← neural net inference (pip)
├── numpy                ← board tensor prep, symmetry transforms (pip)
├── NBoard protocol      ← stdin/stdout text protocol (zero-cost, built-in)
├── CUDA driver          ← GPU inference (system, pre-installed)
├── best.pt checkpoint   ← trained weights (your data/ directory)
├── book.pkl             ← opening book (your data/ directory)
└── Edax binary          ← ONLY during evaluation, not competition (external download)
```

Everything the agent needs at competition time is either a pip package, a file in your `data/` directory, or a protocol you implement yourself in a few dozen lines of Python. There is no web API call, no authentication token, no network request at inference time. The agent is entirely self-contained once the model has been trained.