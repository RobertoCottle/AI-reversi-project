import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    #Standard residual block: two 3x3 convolutions with a skip connection.
    #Batch normalization after each conv, ReLU activation throughout.
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class PolicyValueNet(nn.Module):
    #The shared ResNet body with two output heads.

    #Input:  (batch, 6, 8, 8) board tensor
    #Output: policy logits (batch, 65), value scalar (batch, 1)

    #The 65 policy outputs cover all 64 squares plus one pass action.
    #Policy is returned as raw logits — apply softmax externally when
    #needed for probability distributions.

    #Architecture:
    #    - Input conv: 6 → channels, 3x3
    #    - N residual blocks
    #    - Policy head: 1x1 conv → flatten → linear → 65
    #    - Value head:  1x1 conv → flatten → linear → linear → tanh

    def __init__(self, channels: int = 128, n_blocks: int = 6):
        super().__init__()

        # Input convolution — projects 6 planes to `channels` feature maps
        self.input_conv = nn.Conv2d(6, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(n_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = nn.Linear(32 * 8 * 8, 65)

        # Value head
        self.value_conv  = nn.Conv2d(channels, 16, kernel_size=1, bias=False)
        self.value_bn    = nn.BatchNorm2d(16)
        self.value_fc1   = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2   = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        # Shared body
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)           # raw logits, shape (batch, 65)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))   # scalar in (-1, 1)

        return p, v

    def predict(self, board_tensor: torch.Tensor, device: torch.device):
        #Single-position inference used by MCTS.

        #Args:
        #    board_tensor: (6, 8, 8) tensor from encoder.encode_board()
        #    device:       torch.device to run on

        #Returns:
        #    policy: numpy array of shape (65,), valid probabilities
        #    value:  float scalar

        self.eval()
        with torch.no_grad():
            x = board_tensor.unsqueeze(0).to(device)   # add batch dim
            logits, v = self.forward(x)
            policy = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            value  = v.squeeze().item()
        return policy, value