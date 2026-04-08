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
(We take MuZero concepts such as Reanalyze, we did not use MuZero)
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