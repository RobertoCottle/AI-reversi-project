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