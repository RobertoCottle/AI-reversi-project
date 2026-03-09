02/26/26 Notes:

1. Current implementation involves using a Phase-Based Evaluation and Alpha-Beta Pruning. To elevate this to a "Strong Opponent" level that understands high-level Reversi concepts like Stoner Traps and Parity, you need to move beyond simple weighted sums and into Structural Analysis.

a. Implement Stability analysis
    i. In Reversi, a stone is "stable" if it can never be flipped for the rest of the game. Your current POS_WEIGHTS are static, but stability is dynamic.
        a. Start from the corners, any stone in a corner is stable, any stone adjacenet to a staone stone of the same color cannot be outflanked is also stable.
        b. Create a get_stable_count() function. Stable stones should be weighted significantly higher than "active" stones because they represent guaranteed points and anchor points for future moves.

b. Advanced Edge & Corner Tactics
    i. Code currentl views X-Squares (diagnoal neighbors of corners) as static penalities. Make them Context-Aware:
        a. Stoner Traps: Detect "weak edges". If the opponent occupies the B2 X-square, they may be forcing you to take a move that eventually surrenders the corner.
        b. Wedges: Add a bonus for moves that play between two opponent stones on an edge. This "wedging" prevents the opponent from ever reclaiming that segment.
        c. C-Squares and X-Squares: If a corner is already occupied by you, the penalty for the adjacent C/X squares should vanish or become a bonus, as they are no longer "dangerous" to hold.

c. Sophisticated Mobility (Evaporation)
    i. We hav basic mobility diff but professional Reversi AI uses the Evaporation Stategy in the early game.
        a. The Logic: The goal is to have fewer pieces than your opponent in the first 20 moves while maintaining high mobility. This keeps your "frontier" small and forces the opponent to make moves that open up the board for you
        b. The Improvement: In your early_game phase, flip the piece_diff weight to be negative. You want to minimize your own stone count while maximizing your move options.

d. Parity Management
    i. Parity is the advantage of having the last move in a region (usually an empty area of the board).
        a. The Logic: In the late game, the player who moves last in an empty "hole" usually gains more stones.
        b. The Improvement: In your evaluation function, check the number of empty squares in localized regions. If a region has an even number of squares, and it's the opponent's turn, you have "parity." Add a bonus for states where you are likely to make the final move of the game (usually the player who moves on turn 60).

e. Search Optimizations
    i. search needs to be faster to maintain depth
        a. Transposition Table (TT): Use a hash map (Zobrist Hashing) to store board states you’ve already evaluated. Reversi has many paths to the same board; don't calculate the same position twice.
        b. Iterative Deepening: Instead of a fixed DEPTH = 3, search depth 1, then 2, then 3, etc., until your time limit is almost up. This ensures you always have the best possible move ready.
        c. Null Move Pruning: If a position is so good that even "passing" (giving the opponent two moves in a row) still results in a win, you can prune that branch early.

Other ideas to explore:
f. Aspiration Windows alpha-beta: 
    narrows the alpha-beta window based on previous iteration's score, faster but needs re-search on fail

g. Reversi Pattern based evaliuation: lookup tables for edge patterns, used in Logistello (the strongest Reversi engine ever built)

chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.cs.umd.edu/sites/default/files/scholarly_papers/Engel.pdf


--For short games such as tic-tac-toe, the Minimax algorithm can be used exactly as described, but for longer games like Reversi, chess, Go, etc, such a calculation is only feasible for board positions near the endgame. Because the computational cost is exponential in the search depth, a full search is impractical for most board positions. The runtime of the Minimax algorithm can be improved using alpha-beta pruning [1], but the overall exponential dependence on the search depth cannot be avoided. Because of this, Minimax is usually run with a xed search depth, resulting in leaf nodes which are not endgame boards. The exact score for these boards is therefore unknown, and a board position evaluator must be used to predict the score. Much of the work in game AI centers on creating a good position evaluator- nding important features and designing or training a model to predict the quality of intermediate board positions.

--ProbCut- an algorithm which prunes subtrees which are probablistically irrelevant

--SVM classifer, as well as evaluating intermediate board positions using linear interpolation

-minmax should be used more towards the endgame for searching, because the computational cost is exponential in the search depth, a full search is impractical for most board positions. The runtime of the Minimax algorithm can be improved using alpha-beta pruning [1], but the overall exponential dependence on the search depth cannot be avoided. Because of this, Minimax is usually run with a xed search depth, resulting in leaf nodes which are not endgame boards. The exact score for these boards is therefore unknown, and a board position evaluator must be used to predict the score. Much of the work in game AI centers on creating a good position evaluator- nding important features and designing or training a model to predict the quality of intermediate board positions.

-uses the Minimax algorithm to propagate board evaluations backwards from the endgame.
-SVM classifier

-have minimax learn from the endgame to early game ?

https://www.geeksforgeeks.org/dsa/minimax-algorithm-in-game-theory-set-5-zobrist-hashing/ 
https://samsoft.org.uk/reversi/strategy.htm 
https://samsoft.org.uk/reversi/openings.htm

Min-Max vs. Monte Carlo Tree Search (MCTS)
https://www.geeksforgeeks.org/artificial-intelligence/mini-max-algorithm-in-artificial-intelligence/ 

--3/1/26

-need to update `reversi_tournamentV2` to show the avg corners for the first and second player
===== PHASE AVERAGE CORNERS =====

Early Phase:
  Avg White Corners: 0.00
  Avg Black Corners: 0.00

Mid Phase:
  Avg White Corners: 0.00
  Avg Black Corners: 0.00

Late Phase:
  Avg White Corners: 2.50
  Avg Black Corners: 1.50

  -- 3/2/26

  for `Better_PlayerV3.py` I should not use arrays, as the iteration is slow, I should use a vectorized or bitwise approach to achieve these results.

  -`Better_PlayerV3.py` is now taking to long to make a decision, going past 200 seconds. Start at `Better_PlayerV2.py` and research how to make the algorithm better. Perhaps bitwise or more vector operations. Try to research this part on google mainly, use the ai tools to research descriptions. 

  --3/3/26

-Better_PlayerV3 is able to defeat Better_Player at depth of 7, at depth of 6, it doubles the avg winning discs and corners in wins. however avg decision making time at depth 6+ is 15+ seconds plus, even reaching past a minute

(These are the tournament results with both of a base of 3)
  ===== TOURNAMENT RESULTS =====

Better_Player:
  Wins: 10
  Losses: 10
  Avg Winning Discs: 37.00
  Avg Losing Discs: 24.00
  Avg Corners in Wins: 1.00

Better_PlayerV3:
  Wins: 10
  Losses: 10
  Avg Winning Discs: 40.00
  Avg Losing Discs: 27.00
  Avg Corners in Wins: 2.00

(These are the tournament results with both of a base of 3)
  ===== TOURNAMENT RESULTS =====

Better_PlayerV2:
  Wins: 50
  Losses: 50
  Avg Winning Discs: 37.00
  Avg Losing Discs: 27.00
  Avg Corners in Wins: 1.00

Better_PlayerV3:
  Wins: 50
  Losses: 50
  Avg Winning Discs: 37.00
  Avg Losing Discs: 27.00
  Avg Corners in Wins: 1.00

-Better_PlayerV2 and Better_PlayerV3 do not offer marginal improvements, more adjustments are to be made for further improvements.

--- Donald 3/4/26 -----

a. With added C_square, X_square, wedge control and zobrist hashing `Better_PlayerV3.` beats `Better_Player` at base depth of 3

===== TOURNAMENT RESULTS =====

Better_Player:
  Wins: 0
  Losses: 20
  Avg Winning Discs: 0.00
  Avg Losing Discs: 26.50
  Avg Corners in Wins: 0.00

Better_PlayerV3:
  Wins: 20
  Losses: 0
  Avg Winning Discs: 37.50
  Avg Losing Discs: 0.00
  Avg Corners in Wins: 3.50

b. `Better_PlayerV3.` also beats `Better_PlayerV2` at base depth of 3

===== TOURNAMENT RESULTS =====

Better_PlayerV2:
  Wins: 0
  Losses: 20
  Avg Winning Discs: 0.00
  Avg Losing Discs: 25.00
  Avg Corners in Wins: 0.00

Better_PlayerV3:
  Wins: 20
  Losses: 0
  Avg Winning Discs: 39.00
  Avg Losing Discs: 0.00
  Avg Corners in Wins: 3.00


  -3/8/2026

  Implement these into `Better_Player_V4` for further advancement:

  -Iterative Deepening Depth First Search (IDDFS) Reversi — lets you use time as your budget instead of fixed depth, always has a best move ready if time runs out

  -Reversi pattern based evaluation — lookup tables for edge patterns, used in Logistello (the strongest Reversi engine ever built)

  -Reversi perfect endgame solver — when empties drop below ~14-16, you can solve perfectly with pure alpha-beta since the search space collapses

  -Othello opening book — precomputed first ~10 moves, skips search entirely for known strong openings

  -Architecture: full stability[CHECK] → endgame solver → iterative deepening[CHECK] → bitboards. Bitboards are the biggest rewrite but unlock the most speed.

  -possible to get the minimax to learn the game starting from the ending positions ? 

  -- 3/9/2026

  ===== TOURNAMENT RESULTS =====

(Better PlayerV3 vs BPP_V4-5)
Better_PlayerV3:
  Wins: 0
  Losses: 20
  Avg Winning Discs: 0.00
  Avg Losing Discs: 19.50
  Avg Corners in Wins: 0.00

BP_V4-5:
  Wins: 20
  Losses: 0
  Avg Winning Discs: 44.50
  Avg Losing Discs: 0.00
  Avg Corners in Wins: 2.50

-- BP_V4-5 increases time duration against greedy_player, but does not show it against greedy_player

--After lowering threshold, the issue seems to have resolved. 