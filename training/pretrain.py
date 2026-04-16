import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from network.q_network import QNetwork
from data.demo_buffer import DemoBuffer


MARGIN = 0.8   # minimum Q-value gap between expert move and others


def supervised_loss(
    q_values:      torch.Tensor,
    expert_actions: torch.Tensor,
    margin:        float = MARGIN,
) -> torch.Tensor:
    """
    Large margin supervised loss from DQfD paper.

    Forces Q(s, expert_action) >= Q(s, other_action) + margin
    for all non-expert actions.

    This ensures the expert moves always rank highest
    by at least `margin` above any other move.
    """
    batch_size = q_values.size(0)
    device     = q_values.device

    # Q-value of expert action
    expert_q = q_values.gather(
        1, expert_actions.unsqueeze(1)
    ).squeeze(1)

    # Margin matrix: margin for all actions except expert
    margin_matrix = torch.full_like(q_values, margin)
    margin_matrix.scatter_(1, expert_actions.unsqueeze(1), 0.0)

    # L = max over all a of (Q(s,a) + margin(a)) - Q(s, expert)
    loss = (q_values + margin_matrix).max(dim=1).values - expert_q
    return loss.clamp(min=0).mean()


def run_pretrain(
    demo_path:    str   = "saved/demos/expert_demos.npz",
    save_path:    str   = "saved/models/pretrained_q.pt",
    log_path:     str   = "saved/pretrain_log.csv",
    resume_path:  str   = None,
    epochs:       int   = 20,
    batch_size:   int   = 512,
    lr:           float = 1e-3,
    channels:     int   = 128,
    n_blocks:     int   = 6,
    log_every:    int   = 50,
):
    """
    Phase 2: Pre-train Q-network on expert demonstrations.

    Combines:
        - Supervised large margin loss (expert moves rank highest)
        - TD loss (Bellman equation consistency)

    Supports Ctrl+C and resume.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path),  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Pretrain] Using device: {device}")

    # Load demo buffer
    buffer = DemoBuffer(demo_path)
    if not buffer.load():
        raise FileNotFoundError(
            f"No demo data at {demo_path}. "
            f"Run generate_demos.py first."
        )

    print(f"[Pretrain] Loaded {len(buffer):,} expert transitions")

    # Build tensors from buffer
    boards      = torch.tensor(np.stack(buffer.boards),      dtype=torch.float32)
    actions     = torch.tensor(np.array(buffer.actions),     dtype=torch.long)
    rewards     = torch.tensor(np.array(buffer.rewards),     dtype=torch.float32)
    next_boards = torch.tensor(np.stack(buffer.next_boards), dtype=torch.float32)
    dones       = torch.tensor(np.array(buffer.dones),       dtype=torch.float32)

    dataset = TensorDataset(boards, actions, rewards, next_boards, dones)
    loader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True
    )

    # Initialize or load model
    if resume_path and os.path.exists(resume_path):
        model = QNetwork.load(resume_path, device)
        print(f"[Pretrain] Resuming from {resume_path}")
    else:
        model = QNetwork(channels=channels, n_blocks=n_blocks).to(device)
        print(f"[Pretrain] Starting fresh model")

    # Load target network for TD updates
    target_net = QNetwork(channels=channels, n_blocks=n_blocks).to(device)
    target_net.load_state_dict(model.state_dict())
    target_net.eval()

    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    td_loss_fn = nn.MSELoss()

    # CSV log
    log_exists = os.path.exists(log_path)
    log_file   = open(log_path, 'a', newline='')
    log_writer = csv.DictWriter(log_file, fieldnames=[
        'epoch', 'batch', 'td_loss', 'supervised_loss',
        'total_loss', 'elapsed_min'
    ])
    if not log_exists:
        log_writer.writeheader()

    best_loss   = float('inf')
    train_start = time.time()

    print(f"[Pretrain] Training {epochs} epochs, "
          f"{len(dataset):,} transitions, batch={batch_size}")
    print("-" * 60)

    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()

            total_td  = 0.0
            total_sup = 0.0
            n_batches = 0

            for batch_idx, (b, a, r, nb, d) in enumerate(loader):
                b  = b.to(device)
                a  = a.to(device)
                r  = r.to(device)
                nb = nb.to(device)
                d  = d.to(device)

                # Current Q-values
                q_current = model(b)

                # TD target using target network
                with torch.no_grad():
                    q_next   = target_net(nb).max(dim=1).values
                    td_target = r + 0.99 * q_next * (1 - d)

                # TD loss on expert action Q-values
                q_expert  = q_current.gather(1, a.unsqueeze(1)).squeeze(1)
                loss_td   = td_loss_fn(q_expert, td_target)

                # Supervised margin loss
                loss_sup  = supervised_loss(q_current, a)

                loss = loss_td + loss_sup

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_td  += loss_td.item()
                total_sup += loss_sup.item()
                n_batches += 1

                if (batch_idx + 1) % log_every == 0:
                    elapsed = (time.time() - train_start) / 60
                    log_writer.writerow({
                        'epoch':           epoch + 1,
                        'batch':           batch_idx + 1,
                        'td_loss':         loss_td.item(),
                        'supervised_loss': loss_sup.item(),
                        'total_loss':      loss.item(),
                        'elapsed_min':     f"{elapsed:.2f}",
                    })
                    log_file.flush()

            avg_td  = total_td  / n_batches
            avg_sup = total_sup / n_batches
            avg_tot = avg_td + avg_sup

            epoch_time = time.time() - epoch_start
            elapsed    = (time.time() - train_start) / 60
            remaining  = (epochs - epoch - 1) * (epoch_time / 60)

            print(
                f"  Epoch {epoch+1:>3}/{epochs} | "
                f"td={avg_td:.4f} sup={avg_sup:.4f} "
                f"total={avg_tot:.4f} | "
                f"time={epoch_time:.1f}s | "
                f"elapsed={elapsed:.1f}min ETA={remaining:.1f}min"
            )

            # Update target network every epoch
            target_net.load_state_dict(model.state_dict())
            scheduler.step()

            if avg_tot < best_loss:
                best_loss = avg_tot
                model.save(save_path)
                print(f"           Best model saved "
                      f"(loss={best_loss:.4f})")

    except KeyboardInterrupt:
        print(f"\n[Pretrain] Interrupted at epoch {epoch + 1}")
        emergency = save_path.replace('.pt', f'_epoch{epoch+1}_emergency.pt')
        model.save(emergency)
        print(f"[Pretrain] Emergency save: {emergency}")
        print(f"[Pretrain] Resume by running with "
              f"resume_path='{emergency}'")

    finally:
        log_file.close()

    print(f"\n[Pretrain] Complete. Best model: {save_path}")
    return model