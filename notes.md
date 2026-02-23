-what matters right now is figuring out what I can edit and what I can not, the game engine is done for the most part, what mattters now is creating a player ai that can make the best decisions and dominate based on the algorithms and reversi rules you give it and strategy. 

-despite the algorithm for min-max, white always wins, no matter if its brute force or min maxing

White wins more because:
-First-move advantage exists.
-Your evaluation favors early material.
-Your depth is shallow.
-Reversi punishes greedy play.
-Your heuristic is incomplete.

Try:
-Run AI vs AI
-Swap which one uses minimax
-Increase depth

You’ll notice:
-At low depth → White wins a lot
-At higher depth → it evens out
-With strong heuristics → closer to balanced

The first player:
Controls early board structure
Gets early mobility
Forces second player into reactive moves

Since your evaluation is shallow (flip count or depth-limited search),
White often snowballs.

-black wins when I add more depth

-need to add weights to this in the context of the algorithm


--feb,23,2026:
-not just a better algorithm, but the code needs to be optimized, in order to:
1. Save memory space to compute faster, that means optimizing code to be O(n) efficient
2. Implementing the algorithm with heurisitcs for the best strategies as well as most common positions in game
3. Finding where to implement certain algorithms or algorithms in place for early, mid, and end game advancement.
4. ai should also have defensive capabilities and be able to tell what strategy the othe enemy is trying to do.

Possible advancement ideas:
a. Detect true stable discs
b. Detect parity regions
c. Detect edge stability chains
d. Use pattern databases (https://samsoft.org.uk/reversi/strategy.htm)
e. Use exact endgame solver