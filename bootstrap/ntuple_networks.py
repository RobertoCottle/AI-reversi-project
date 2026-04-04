import numpy as np
import pickle
from itertools import product

class NTupleNetwork:
    
    #A lookup-table value function over the Reversi board.

    #The board is a flat 64-element array of {-1, 0, 1}.
    #We define a set of tuples — small groups of board positions —
    #and store one weight per (tuple_index, pattern_value) combination.

    #For a 4-cell tuple, each cell can be in 3 states, so there are
    #3^4 = 81 possible patterns per tuple. With 32 tuples covering
    #all rows, columns, and diagonals, the total parameter count is
    #roughly 32 * 81 = 2,592 weights — extremely compact.

    #Board convention (from reversi.py):
    #    1  = white
    #   -1  = black
    #    0  = empty
    #We shift to {0, 1, 2} internally for array indexing.

    STATES_PER_CELL = 3   # 0=empty, 1=white, 2=black (after shift)

    def __init__(self, tuples: list[list[int]]):
        
        #Args:
        #    tuples: list of tuples, each a list of flat board indices (0–63).
        #            All tuples must have the same length.
        
        self.tuples = tuples
        self.n_tuples = len(tuples)
        self.tuple_length = len(tuples[0])
        self.n_patterns = self.STATES_PER_CELL ** self.tuple_length

        # One weight table per tuple, initialized to zero
        self.weights = [
            np.zeros(self.n_patterns, dtype=np.float32)
            for _ in range(self.n_tuples)
        ]

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shift(cell_value: float) -> int:
        # Map board cell {-1, 0, 1} → index {0, 1, 2}.
        return int(cell_value) + 1

    def _pattern_index(self, flat_board: np.ndarray, tuple_idx: int) -> int:
        
        #Compute the base-3 index for one tuple given the current board.

        #Example: tuple positions [0, 1, 8, 9], values [1, 0, -1, 1]
        #→ shifted  [2, 1, 0, 2]
        #→ index    2*27 + 1*9 + 0*3 + 2*1 = 65
        
        positions = self.tuples[tuple_idx]
        index = 0
        for pos in positions:
            index = index * self.STATES_PER_CELL + self._shift(flat_board[pos])
        return index

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def evaluate(self, board: np.ndarray) -> float:
        
        #Return a scalar value estimate for a board state.

        #Args:
        #    board: 8×8 numpy array as returned by reversi.board
        #Returns:
        #    float in roughly [-1, 1] representing estimated outcome
        #    from white's (piece=1) perspective
        
        flat = board.flatten()
        total = 0.0
        for i in range(self.n_tuples):
            idx = self._pattern_index(flat, i)
            total += self.weights[i][idx]
        return float(np.tanh(total))   # squash to (-1, 1)

    def evaluate_for_player(self, board: np.ndarray, player: int) -> float:
        
        #Return value from the perspective of `player` (1=white, -1=black).
        
        raw = self.evaluate(board)
        return raw if player == 1 else -raw

    # ------------------------------------------------------------------
    # TD weight update
    # ------------------------------------------------------------------

    def update(self, board: np.ndarray, target: float, lr: float = 0.001):
        
        #Apply a TD update: nudge weights toward `target`.

        #The gradient of tanh(sum_of_weights) with respect to any single
        #weight is (1 - tanh²(sum)) = (1 - value²), so the update rule is:

        #    Δw = lr * (target - value) * (1 - value²)

        #Each active tuple receives the same scalar delta.
        
        flat = board.flatten()
        value = self.evaluate(board)
        delta = lr * (target - value) * (1.0 - value ** 2)
        for i in range(self.n_tuples):
            idx = self._pattern_index(flat, i)
            self.weights[i][idx] += delta

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'tuples': self.tuples, 'weights': self.weights}, f)
        print(f"[NTupleNetwork] saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'NTupleNetwork':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        net = cls(data['tuples'])
        net.weights = data['weights']
        print(f"[NTupleNetwork] loaded from {path}")
        return net

    # ------------------------------------------------------------------
    # Factory: standard tuple set for 8×8 Reversi
    # ------------------------------------------------------------------

    @staticmethod
    def make_standard_tuples() -> list[list[int]]:
        
        #Build a systematic set of 4-cell straight tuples covering:
        #  - all 8 rows (8 horizontal 4-tuples each = 40 tuples)
        #  - all 8 columns (40 tuples)
        #  - diagonals of length >= 4 (52 tuples)

        #Returns flat board indices (row*8 + col).
        
        tuples = []
        size = 4   # tuple length

        # Horizontal tuples
        for row in range(8):
            for start_col in range(8 - size + 1):
                t = [row * 8 + (start_col + k) for k in range(size)]
                tuples.append(t)

        # Vertical tuples
        for col in range(8):
            for start_row in range(8 - size + 1):
                t = [(start_row + k) * 8 + col for k in range(size)]
                tuples.append(t)

        # Diagonal (top-left to bottom-right)
        for start_row in range(8 - size + 1):
            for start_col in range(8 - size + 1):
                t = [(start_row + k) * 8 + (start_col + k) for k in range(size)]
                tuples.append(t)

        # Diagonal (top-right to bottom-left)
        for start_row in range(8 - size + 1):
            for start_col in range(size - 1, 8):
                t = [(start_row + k) * 8 + (start_col - k) for k in range(size)]
                tuples.append(t)

        return tuples