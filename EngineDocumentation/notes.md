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