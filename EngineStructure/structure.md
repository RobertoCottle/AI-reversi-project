ExpertQLearning/
│
├── reversi.py
├── reversi_server.py
├── setup.py
│
├── core/
│   ├── __init__.py
│   └── encoder.py            ← same as before
│
├── network/
│   ├── __init__.py
│   └── q_network.py          ← Q-network, outputs 65 Q-values
│
├── data/
│   ├── __init__.py
│   ├── generator.py          ← generates AlphaBeta demo games
│   └── demo_buffer.py        ← stores expert demonstrations
│
├── training/
│   ├── __init__.py
│   ├── pretrain.py           ← supervised pre-training on demos
│   └── dqfd_trainer.py       ← combined TD + supervised fine-tuning
│
├── agents/
│   ├── __init__.py
│   ├── random_agent.py       ← same as before
│   ├── greedy_agent.py       ← same as before
│   ├── alphabeta_agent.py    ← same as before
│   └── competition_agent.py ← uses Q-network directly, no MCTS
│
├── scripts/
│   ├── generate_demos.py     ← Phase 1 entry point
│   ├── pretrain.py           ← Phase 2 entry point
│   ├── finetune.py           ← Phase 3 entry point
│   ├── evaluate.py           ← test against opponents
│   └── compete.py            ← socket client for competition
│
└── saved/
    ├── demos/
    │   └── expert_demos.npz  ← AlphaBeta game data
    └── models/
        ├── pretrained_q.pt   ← after Phase 2
        └── final_q.pt        ← after Phase 3