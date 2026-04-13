import math
import copy
import numpy as np
import torch

from reversi_ai.reversi import reversi as reversi_game
from bootstrap.td_selfplay import get_legal_moves_from_board
from model.encoder import encode_board
from model.network import PolicyValueNet
from search.puct import puct_score


class MCTSNode:
    
    #A single node in the MCTS search tree.

    #Each node represents a board state and stores the statistics
    #needed for PUCT selection and value backpropagation.

    #Attributes:
    #    board:         8x8 numpy array — the board at this node
    #    player:        whose turn it is at this node (1=white, -1=black)
    #    parent:        parent MCTSNode, None for root
    #    action:        (x, y) move that led to this node, None for root
    #    children:      dict mapping (x, y) → MCTSNode
    #    prior:         P(s,a) from policy network, set by parent on expansion
    #    visit_count:   N(s,a) — how many times this node has been visited
    #    value_sum:     W(s,a) — cumulative value from all visits
    #    is_expanded:   whether children have been created yet
    #    is_terminal:   whether this is a game-over position

    def __init__(
        self,
        board: np.ndarray,
        player: int,
        parent: 'MCTSNode' = None,
        action: tuple = None,
        prior: float = 0.0,
    ):
        self.board       = board
        self.player      = player
        self.parent      = parent
        self.action      = action
        self.prior       = prior
        self.children    = {}
        self.visit_count = 0
        self.value_sum   = 0.0
        self.is_expanded = False
        self.is_terminal = False

    @property
    def q_value(self) -> float:
        # Mean action value Q(s,a) = W(s,a) / N(s,a).
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def parent_visits(self) -> int:
        if self.parent is None:
            return self.visit_count
        return self.parent.visit_count

    def best_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        
        #Return the child with the highest PUCT score.
        #Called during the selection phase.
        
        best_score = -np.inf
        best_node  = None
        for child in self.children.values():
            score = puct_score(
                q_value      = child.q_value,
                prior        = child.prior,
                parent_visits= self.visit_count,
                child_visits = child.visit_count,
                c_puct       = c_puct,
            )
            if score > best_score:
                best_score = score
                best_node  = child
        return best_node

    def update(self, value: float):
        
        #Backpropagate a value estimate up through the tree.
        #Value is always from the perspective of the node's player.
        
        self.visit_count += 1
        self.value_sum   += value


class MonteCarloTreeSearch:
    
    #MCTS engine using the PolicyValueNet for leaf evaluation.

    #The four phases per simulation:
    #    1. Selection   — traverse from root using PUCT until a leaf
    #    2. Expansion   — expand the leaf by querying the neural network
    #    3. Evaluation  — use the network's value head as the leaf value
    #    4. Backprop    — propagate value back up to the root

    #Usage:
    #    mcts = MonteCarloTreeSearch(model, device)
    #    policy, value = mcts.search(board, player, n_simulations=400)
    #    action = mcts.select_action(policy, temperature=1.0)
    

    def __init__(
        self,
        model:  PolicyValueNet,
        device: torch.device,
        c_puct: float = 1.5,
    ):
        self.model  = model
        self.device = device
        self.c_puct = c_puct

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        board:         np.ndarray,
        player:        int,
        n_simulations: int = 400,
        add_noise:     bool = False,
    ) -> tuple[np.ndarray, float]:
        
        #Run n_simulations from the given board position.

        #Args:
        #    board:         8x8 numpy array
        #    player:        current player (1=white, -1=black)
        #    n_simulations: number of MCTS simulations to run
        #    add_noise:     add Dirichlet noise to root priors (use during
        
        #                   self-play training, not competition)

        #Returns:
        #    policy: numpy array (65,) — visit count distribution
        #    value:  float — root node value estimate
        
        root = MCTSNode(board.copy(), player)

        # Expand root immediately so it has children to select from
        self._expand(root)

        # Add Dirichlet noise to root priors for training exploration
        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(n_simulations):
            self._simulate(root)

        # Build policy from visit counts
        policy = np.zeros(65, dtype=np.float32)
        for action, child in root.children.items():
            x, y = action
            idx = x * 8 + y
            policy[idx] = child.visit_count

        # Handle pass action
        if not root.children:
            policy[64] = 1.0
        else:
            total = policy.sum()
            if total > 0:
                policy /= total

        return policy, root.q_value

    def select_action(
        self,
        policy:      np.ndarray,
        temperature: float = 1.0,
    ) -> tuple[int, int]:
        
        #Sample an action from the MCTS policy distribution.

        #Args:
        #    policy:      (65,) visit count distribution from search()
        #    temperature: controls how greedy the selection is.
        #                 1.0 = proportional to visits (exploration)
        #                 →0  = always pick most visited (exploitation)

        #Returns:
        #    (x, y) board coordinates, or (-1, -1) for pass
        
        if temperature < 1e-3:
            # Deterministic: pick most visited action
            idx = int(np.argmax(policy))
        else:
            # Sample proportional to visit_count^(1/temperature)
            counts = policy ** (1.0 / temperature)
            total  = counts.sum()
            if total == 0:
                idx = 64   # pass
            else:
                counts /= total
                idx = int(np.random.choice(len(counts), p=counts))

        if idx == 64:
            return (-1, -1)   # pass signal matching server convention
        return (idx // 8, idx % 8)

    # ------------------------------------------------------------------
    # Internal simulation phases
    # ------------------------------------------------------------------

    def _simulate(self, root: MCTSNode):
        # Run one full simulation: select → expand → evaluate → backprop.
        node = root
        path = [node]

        # Phase 1: Selection — traverse until unexpanded or terminal
        while node.is_expanded and not node.is_terminal:
            node = node.best_child(self.c_puct)
            path.append(node)

        # Phase 2 & 3: Expansion and evaluation
        if node.is_terminal:
            # Terminal node: use actual game outcome as value
            value = self._terminal_value(node)
        else:
            # Expand and get value from neural network
            value = self._expand(node)

        # Phase 4: Backpropagation
        # Flip value sign at each level because players alternate
        for node in reversed(path):
            node.update(value)
            value = -value

    def _expand(self, node: MCTSNode) -> float:
        #Expand a leaf node using the neural network.

        #Queries the network for policy priors and value estimate,
        #creates child nodes for all legal moves, and marks node as expanded.

        #Returns the value estimate from the network (from node.player's
        #perspective).
        
        # Encode board and query network
        tensor = encode_board(node.board, node.player)
        policy_probs, value = self.model.predict(tensor, self.device)

        # Find legal moves using the game engine
        sim = reversi_game()
        sim.board = node.board.copy()
        legal = get_legal_moves_from_board(node.board, node.player)

        if not legal:
            # No legal moves — check if opponent can move
            opp_sim = reversi_game()
            opp_sim.board = node.board.copy()
            opp_legal = get_legal_moves_from_board(node.board, node.player)

            if not opp_legal:
                # Neither player can move — terminal state
                node.is_terminal = True
                node.is_expanded = True
                return self._terminal_value(node)
            else:
                # Current player must pass — create one pass child
                pass_child = MCTSNode(
                    board  = node.board.copy(),
                    player = -node.player,
                    parent = node,
                    action = (-1, -1),
                    prior  = 1.0,
                )
                node.children[(-1, -1)] = pass_child
                node.is_expanded = True
                return value

        # Create child nodes for each legal move
        for (x, y) in legal:
            # Simulate the move
            child_sim = reversi_game()
            child_sim.board = node.board.copy()
            child_sim.step(x, y, node.player, commit=True)

            prior = float(policy_probs[x * 8 + y])
            child = MCTSNode(
                board  = child_sim.board.copy(),
                player = -node.player,
                parent = node,
                action = (x, y),
                prior  = prior,
            )
            node.children[(x, y)] = child

        node.is_expanded = True
        return value

    def _terminal_value(self, node: MCTSNode) -> float:
        
        #Compute the actual game outcome at a terminal node.
        #Returns value from node.player's perspective.
        
        sim = reversi_game()
        sim.board = node.board.copy()
        white = int(np.sum(sim.board == 1))
        black = int(np.sum(sim.board == -1))

        if white > black:
            outcome = 1.0    # white wins
        elif black > white:
            outcome = -1.0   # black wins
        else:
            outcome = 0.0    # draw

        # Return from current node's player perspective
        return outcome if node.player == 1 else -outcome

    def _add_dirichlet_noise(self, root: MCTSNode, epsilon: float = 0.25, alpha: float = 0.3):
        
        #Add Dirichlet noise to root node priors during self-play training.
        #This ensures the agent explores moves the policy initially rates low,
        #preventing premature convergence to a narrow set of openings.
        
        n_children = len(root.children)
        noise = np.random.dirichlet([alpha] * n_children)
        for child, eta in zip(root.children.values(), noise):
            child.prior = (1 - epsilon) * child.prior + epsilon * eta