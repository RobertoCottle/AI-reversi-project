import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model.network import PolicyValueNet


class Trainer:
    #Applies gradient updates to the PolicyValueNet using positions
    #sampled from the ReplayStore.

    #Loss function is a sum of two terms:
    #    policy loss: cross-entropy between predicted policy and MCTS policy
    #    value loss:  mean squared error between predicted value and outcome

    #Both heads are trained jointly through the shared residual body,
    #so improvements in value estimation also benefit policy learning.

    def __init__(
        self,
        model:      PolicyValueNet,
        device:     torch.device,
        lr:         float = 2e-3,
        l2_reg:     float = 1e-4,
    ):
        self.model  = model
        self.device = device

        self.optimizer = optim.Adam(
            model.parameters(),
            lr           = lr,
            weight_decay = l2_reg,   # L2 regularization built into Adam
        )

        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn  = nn.MSELoss()

        self.train_history = []   # list of dicts, one per update step

    def update(self, boards, policies, values) -> dict:
        #Apply one gradient update step on a minibatch.

        #Args:
        #    boards:    np.ndarray (batch, 6, 8, 8)
        #    policies:  np.ndarray (batch, 65)
        #    values:    np.ndarray (batch, 1)

        #Returns:
        #    dict with loss breakdown for logging
        
        self.model.train()

        # Move to device
        board_t  = torch.tensor(boards,   dtype=torch.float32).to(self.device)
        policy_t = torch.tensor(policies, dtype=torch.float32).to(self.device)
        value_t  = torch.tensor(values,   dtype=torch.float32).to(self.device)

        # Forward pass
        policy_logits, value_pred = self.model(board_t)

        # Compute losses
        p_loss = self.policy_loss_fn(policy_logits, policy_t)
        v_loss = self.value_loss_fn(value_pred, value_t)
        loss   = p_loss + v_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        result = {
            'loss':         loss.item(),
            'policy_loss':  p_loss.item(),
            'value_loss':   v_loss.item(),
        }
        self.train_history.append(result)
        return result

    def run_update_steps(self, store, n_steps: int, batch_size: int) -> dict:
        
        #Run n_steps gradient updates sampling from the replay store.

        #Returns averaged loss metrics across all steps.
        
        total = {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}

        for _ in range(n_steps):
            boards, policies, values = store.sample(batch_size)
            metrics = self.update(boards, policies, values)
            for k in total:
                total[k] += metrics[k]

        return {k: v / n_steps for k, v in total.items()}

    def save_checkpoint(self, path: str, iteration: int, channels: int, n_blocks: int):
        # Save model weights and training metadata.
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state':  self.optimizer.state_dict(),
            'iteration':        iteration,
            'channels':         channels,
            'n_blocks':         n_blocks,
        }, path)

    @staticmethod
    def load_checkpoint(path: str, device: torch.device):
        
        #Load a checkpoint and return the model and iteration number.
    
        checkpoint = torch.load(path, map_location=device)
        model = PolicyValueNet(
            channels = checkpoint['channels'],
            n_blocks = checkpoint['n_blocks'],
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint.get('iteration', 0)