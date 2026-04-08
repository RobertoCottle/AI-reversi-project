Construct an AI for Reversi using only non-deep learning methods that interacts with the provided environment (See the codes under the Files section)

Rules:

1. Standard Reversi rules (Check https://en.wikipedia.org/wiki/ReversiLinks to an external site.), except white goes first instead of black.
2. Two games per round, you go first in one game, your opponent goes first in the other
The combined piece lead determines the result.
    a. e.g. you got 36-28 (+8) in the first game, 25 - 39 (-14) in the second, then the combined lead is 8-14 = -6 and thus you lose.
3. Code must be written in Python
4. No multi-threading
5. Cannot be the exact same algorithm with your Project 1
6. less then 1,000,000 total trainable parameters (All networks combined if you are using multiple, e,g actor-critic)
7. No more then 5 seconds per hand.
8. The codes will be ran on my laptop plugged in during the tournament

-use the same reversi.py and reversi_server.py 
-use deep learning methods

DEEP LEARNING METHODS With reversi:

1. double q learning?
2. monte carlo tree and neural networks
3. MCTS with neural network  ?
4. AlphaZero-style Reinforcement Learning (MCTS + Neural Network):
--- This is considered the state-of-the-art. It combines neural networks to evaluate board positions and suggest moves (policy/value heads) with MCTS to simulate future moves. This approach trains by playing millions of games against itself to maximize winning probability.

5. Deep Q-Learning (DQN): Deep neural networks are used as function approximators to estimate the Q-value (expected total reward) of different moves in a given board state

6. Multilayer Perceptron Networks (MLP): MLP networks are effective in acting as strong evaluation functions (replacing traditional heuristics) when paired with searching algorithms, often finding the highest level of play compared to simpler methods

AlphaZero-style with full MCTS + a deep residual network is the state of the art when compute is available. The key properties you want are deep lookahead at inference time and a value network trained on a rich diversity of positions. The modifications that help most here are Go-Exploit-style exploration (so the value network is accurate across all game phases), and the MCTS + n-tuple hybrid (for lighter hardware). Model-based methods like AlphaZero tend to be more sample-efficient than model-free methods at the cost of greater computational complexity during runtime. arXiv
Against other AIs specifically, deeper MCTS search at inference time is the single largest lever — more simulations per move directly translates to stronger play in Reversi because the game has clear positional features (corners, mobility) that MCTS reliably discovers with enough lookahead.

The three improvement categories explained
1. Better exploration (where to start self-play)
Go-Exploit samples the start state of self-play trajectories from an archive of states of interest. Beginning trajectories from varied starting states enables it to more effectively explore the game tree and learn a value function that generalizes better. Producing shorter trajectories allows training on more independent value targets. In Connect Four and 9×9 Go, Go-Exploit learns with greater sample efficiency than standard AlphaZero, resulting in stronger performance. Medium
A related strategy addresses the cold-start problem from a different angle. Researchers embed MCTS enhancements like Rollout and RAVE at the start period of iterative self-play training. The hypothesis is that since the neural network and MCTS statistics are initialized randomly, warm-starting with these enhancements can cure the cold-start problem — and experiments on 6×6 Othello found training with these enhancements was mostly better than the baseline. Scholarly Publications
Another approach proposes an improved self-play data generation strategy that initially emphasizes latter game phases and gradually extends to entire games as training progresses. In Connect4 and Breakthrough, agents using this improved approach learn significantly faster than counterpart agents using the standard approach. ResearchGate
2. Better training signal (path consistency and reanalysis)
PCZero augments AlphaZero's training by minimizing violations of the "path consistency" condition — the principle that values on one optimal path should be identical. PCZero's learning curve grows faster and maintains higher Elo ratings than standard AlphaZero. Proceedings of Machine Learning Research
MuZero introduced a "Reanalyze" mode optimized for sample efficiency: it reanalyzes old trajectories by re-running MCTS using the latest network parameters to provide fresh targets. When applied to 57 Atari games, MuZero Reanalyze achieved dramatically better median scores than previous model-free approaches including Rainbow and IMPALA. arXiv
3. Better architecture
Scheiermann and Konen wrapped MCTS around RL n-tuple networks (keeping the costly MCTS completely out of training and using it only at test time), creating agents trainable in less than 2 hours on a single standard CPU. This architecture is the first trained on standard hardware to beat the Othello program Edax up to and including level 7, where most other algorithms could only defeat Edax up to level 2. ResearchGate
ScalableAlphaZero combines a graph neural network with AlphaZero to enable incremental learning: by training for only three days on small Othello boards, it can defeat the standard AlphaZero model trained on the large board for 30 days

REFERENCES:
https://www.researchgate.net/publication/369876515_Application_of_Different_Artificial_Intelligence_Methods_on_Reversi 

https://pmc.ncbi.nlm.nih.gov/articles/PMC9137882/

common commands:

type nul > bootstrap\__init__.py (to create init files)

Layer 2 — CPU only, no GPU needed
Layer 3 pretraining — GPU will make this significantly faster
Layer 4 MCTS — GPU helps during self-play data generation but not required
Layer 5/6 training loop — GPU is where most of the speedup comes from
Layer 7 evaluation — CPU is fine, you're not in a hurry
Layer 8 competition — CPU is sufficient for inference

why layer 5 is needed:

Layer 5 is needed because Layer 6 (the training engine) cannot train on nothing — it needs actual game data to learn from, and that data has to come from the agent playing against itself.

The key insight is that the pretrained network from Layer 3 was only trained on n-tuple value estimates, which are rough approximations. It has never actually played a game using MCTS search. Layer 5 bridges that gap by taking the pretrained network, running it through real games with real MCTS search, and recording every position it sees along with two critical pieces of information that Layer 3 never had:

The MCTS policy — not just which moves are legal, but which moves MCTS actually considered good after 400 simulations of lookahead. This is a much richer training signal than the uniform-over-legal-moves policy that Layer 3 used.

The actual game outcome — whether the position ultimately led to a win or loss. This grounds the value estimates in real results rather than n-tuple approximations.

Without Layer 5, Layer 6 has no data to train on. Without Layer 6 improving the network, Layer 5 would keep generating data from a weak network forever. The two layers are designed to alternate — Layer 5 generates fresh game data from the current best network, Layer 6 trains a stronger network from that data, then Layer 5 runs again with the stronger network to generate better data, and so on. That iterative loop is the entire mechanism by which the agent improves beyond the pretrained starting point.

In short: Layer 3 gives the network a warm start, Layer 5 gives Layer 6 something real to train on.

pip install -e . -use after everytime you create a new folder, and put in a new __init__.py file.


layer 5 output:
Game  190/200 | Buffer: 11,634 | W=42.11% B=46.32% D=11.58%
Game  200/200 | Buffer: 12,248 | W=42.00% B=46.50% D=11.50%

layer 6- 

Layer 6 is the heart of AlphaZero — the iterative improvement loop where the agent genuinely gets stronger over time. Let me walk through exactly what is happening.

---

## The core idea

AlphaZero is built on one insight: a neural network can teach itself to play better by playing against itself, using its own search process to generate higher quality training targets than it currently produces. The network starts imperfect, plays games, the search process produces better decisions than the raw network would alone, and those better decisions become the training signal that improves the network. Then the stronger network produces even better search, which produces even better training data, and so on. This is the loop.

---

## What happens in each iteration

Every iteration has exactly two phases that alternate.

### Phase 1 — Self-play (generating data)

The `SelfPlayWorker` plays 20 complete games from the opening position. For every single move in every game, it does not just ask the network "what move do you recommend?" — instead it runs 400 MCTS simulations, which means it looks ahead through hundreds of possible continuations using the network to evaluate leaf positions. The result of those 400 simulations is a visit count distribution across all legal moves — the moves that MCTS explored most heavily get the highest visit counts.

This MCTS policy is meaningfully stronger than the raw network policy for a subtle reason. The network alone sees the current position and outputs a guess. The MCTS sees the current position, simulates 400 futures, and outputs a distribution shaped by what actually worked in those futures. It is the difference between guessing and calculating. That calculated distribution becomes the policy target stored in the replay buffer.

After the game ends with a winner, every position in that game gets labeled with the actual outcome — win, loss, or draw from the perspective of whoever was to move in that position. That outcome becomes the value target stored in the replay buffer.

So after Phase 1 of one iteration, roughly 20 × 60 = 1,200 new positions have been added to the buffer, each carrying a high-quality policy target from MCTS and a ground-truth value target from the actual game result.

### Phase 2 — Gradient updates (improving the network)

The `Trainer` samples 100 random minibatches of 256 positions from the replay buffer and runs backpropagation on each one. For each minibatch it computes two losses simultaneously:

The policy loss is cross-entropy between what the network currently predicts as the best moves and what the MCTS search actually found to be the best moves. If the network says move A has 10% probability but MCTS visited move A 60% of the time, the policy loss is high and the gradient pushes the network to assign move A higher probability in future. Over many updates the network learns to internalize what MCTS knows — it learns to predict good moves directly without needing to search.

The value loss is mean squared error between what the network predicts the outcome will be and what the outcome actually was. If the network says a position is slightly winning but the game was actually lost, the gradient pushes the value estimate downward for similar positions. Over many updates the network learns accurate position evaluation.

Both losses flow backward through the same shared residual tower, so every gradient update improves both the value estimates and the move recommendations simultaneously.

---

## Why the loop produces improvement

The reason this converges to genuine strength rather than just spinning in circles is that the MCTS search acts as a policy improvement operator. Even with an imperfect network, running 400 simulations consistently finds better moves than the raw network predicts. When those better moves become training targets, the network gets pulled toward the search's understanding. The next iteration's search then starts from a stronger network and finds even better moves. Each cycle the gap between what the network knows and what the search knows narrows, and the overall level of play rises.

By iteration 200 with 20 games per iteration, the network has seen roughly 200 × 20 × 60 = 240,000 positions from its own play on top of the pretrained starting point, each labeled with MCTS-quality policy targets and real game outcomes. The network has been nudged toward better move evaluation 200 × 100 = 20,000 gradient steps. The `best.pt` checkpoint saved at the end is the distilled product of all of that — a network that can produce strong move recommendations in a single forward pass without any search at all, though it plays even stronger when MCTS is wrapped around it at competition time.


### layer 6 results

  Iter  194/200 | Buf=243,002 | W=50.00% B=40.00% D=10.00% | loss=2.0852 (p=1.7449 v=0.3403) | time=305.0s
  Iter  195/200 | Buf=244,229 | W=40.00% B=60.00% D=0.00% | loss=2.0961 (p=1.7512 v=0.3450) | time=351.2s
  Iter  196/200 | Buf=245,454 | W=30.00% B=60.00% D=10.00% | loss=2.0929 (p=1.7446 v=0.3483) | time=325.1s
  Iter  197/200 | Buf=246,680 | W=20.00% B=75.00% D=5.00% | loss=2.0869 (p=1.7358 v=0.3511) | time=357.4s
  Iter  198/200 | Buf=247,908 | W=35.00% B=60.00% D=5.00% | loss=2.1041 (p=1.7549 v=0.3492) | time=344.4s
  Iter  199/200 | Buf=249,133 | W=35.00% B=50.00% D=15.00% | loss=2.0886 (p=1.7405 v=0.3482) | time=339.0s
  Iter  200/200 | Buf=250,356 | W=60.00% B=35.00% D=5.00% | loss=2.0953 (p=1.7492 v=0.3461) | time=301.9s
           Checkpoint saved → data/checkpoints/iter_0200.pt
[ReplayStore] Saved 250,356 examples to data/replay_buffer/buffer.npz