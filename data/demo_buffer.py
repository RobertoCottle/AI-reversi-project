import os
import numpy as np


class DemoBuffer:
    """
    Stores expert demonstration transitions.

    Each transition is:
        board:       (6, 8, 8) float32 — board state
        action:      int64 — flat move index (x*8+y or 64 for pass)
        reward:      float32 — immediate reward (0 during game, +1/-1 at end)
        next_board:  (6, 8, 8) float32 — board after move
        done:        bool — whether game ended
        is_expert:   bool — always True for demo buffer
    """

    def __init__(self, save_path: str = "saved/demos/expert_demos.npz"):
        self.save_path  = save_path
        self.boards      = []
        self.actions     = []
        self.rewards     = []
        self.next_boards = []
        self.dones       = []
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def push(
        self,
        board:      np.ndarray,
        action:     int,
        reward:     float,
        next_board: np.ndarray,
        done:       bool,
    ):
        self.boards.append(board.astype(np.float32))
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_boards.append(next_board.astype(np.float32))
        self.dones.append(done)

    def __len__(self):
        return len(self.boards)

    def sample(self, batch_size: int) -> dict:
        n       = len(self.boards)
        indices = np.random.choice(n, min(batch_size, n), replace=False)
        return {
            'boards':      np.stack([self.boards[i]      for i in indices]),
            'actions':     np.array([self.actions[i]     for i in indices]),
            'rewards':     np.array([self.rewards[i]     for i in indices]),
            'next_boards': np.stack([self.next_boards[i] for i in indices]),
            'dones':       np.array([self.dones[i]       for i in indices]),
        }

    def save(self):
        if not self.boards:
            print("[DemoBuffer] Empty — nothing to save")
            return
        np.savez_compressed(
            self.save_path,
            boards      = np.stack(self.boards),
            actions     = np.array(self.actions),
            rewards     = np.array(self.rewards),
            next_boards = np.stack(self.next_boards),
            dones       = np.array(self.dones),
        )
        print(f"[DemoBuffer] Saved {len(self.boards):,} transitions "
              f"to {self.save_path}")

    def load(self) -> bool:
        if not os.path.exists(self.save_path):
            print("[DemoBuffer] No existing demos found")
            return False
        data             = np.load(self.save_path)
        self.boards      = list(data['boards'])
        self.actions     = list(data['actions'])
        self.rewards     = list(data['rewards'])
        self.next_boards = list(data['next_boards'])
        self.dones       = list(data['dones'])
        print(f"[DemoBuffer] Loaded {len(self.boards):,} transitions "
              f"from {self.save_path}")
        return True