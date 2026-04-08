import math


def puct_score(
    q_value: float,
    prior: float,
    parent_visits: int,
    child_visits: int,
    c_puct: float = 1.5,
) -> float:
    
    #PUCT selection score used at every node in the MCTS tree.

    #Score = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    #Where:
    #    Q(s,a)    = mean action value (exploitation)
    #    P(s,a)    = prior probability from policy network (exploration guide)
    #    N(s)      = total visit count of parent node
    #    N(s,a)    = visit count of this child
    #    c_puct    = exploration constant, typically 1.5-2.0 for Reversi

    #When child_visits is 0 the score is dominated by the prior,
    #encouraging the agent to try unvisited promising moves first.
    #When child_visits is high, Q dominates and the agent exploits
    #what it has already learned about that branch.
    
    exploration = c_puct * prior * math.sqrt(parent_visits) / (1 + child_visits)
    return q_value + exploration



# The PUCT (Predictor Upper Confidence bounds applied to Trees) score is a selection strategy used in Monte Carlo Tree Search (MCTS), 
# popularized by DeepMind's AlphaZero. It balances exploitation (choosing moves known to be good) with 
# exploration (investigating moves with high potential or few visits)