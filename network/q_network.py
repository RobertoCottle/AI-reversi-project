import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.encoder import encode_board


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class QNetwork(nn.Module):
    """
    Deep Q-Network for Reversi.

    Input:  (batch, 6, 8, 8) board tensor
    Output: (batch, 65) Q-values — one per square plus pass action

    At inference time pick argmax over legal moves only.
    """

    def __init__(self, channels: int = 128, n_blocks: int = 6):
        super().__init__()
        self.channels = channels
        self.n_blocks  = n_blocks

        self.input_conv = nn.Conv2d(6, channels, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(n_blocks)
        ])

        self.q_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.q_bn   = nn.BatchNorm2d(32)
        self.q_fc   = nn.Linear(32 * 8 * 8, 65)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        x = F.relu(self.q_bn(self.q_conv(x)))
        x = x.view(x.size(0), -1)
        return self.q_fc(x)

    def predict_move(
        self,
        board:   np.ndarray,
        player:  int,
        legal:   list,
        device:  torch.device,
    ) -> tuple:
        """
        Select best legal move using Q-values.
        Returns (x, y) or (-1, -1) for pass.
        """
        if not legal:
            return -1, -1

        self.eval()
        with torch.no_grad():
            tensor   = encode_board(board, player)
            q_values = self.forward(
                tensor.unsqueeze(0).to(device)
            ).squeeze(0)

        best_q    = -float('inf')
        best_move = legal[0]

        for x, y in legal:
            q = q_values[x * 8 + y].item()
            if q > best_q:
                best_q    = q
                best_move = (x, y)

        return best_move

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'channels':         self.channels,
            'n_blocks':         self.n_blocks,
        }, path)
        print(f"[QNetwork] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> 'QNetwork':
        ckpt  = torch.load(path, map_location=device)
        model = cls(
            channels = ckpt['channels'],
            n_blocks = ckpt['n_blocks'],
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[QNetwork] Loaded from {path}")
        return model