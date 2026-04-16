[DemoBuffer] Loaded 15,840 transitions from saved/demos/expert_demos.npz
[Generator] Resuming from ~264 games
[Generator] Generating 1736 games (AlphaBeta-d3 vs AlphaBeta-d3)
[Generator] Time limit: 1.0s per move
[Generator] Save every: 50 games
------------------------------------------------------------
  Game   314/2000 | W=2.0% B=98.0% D=0.0% | trans=18,840 | elapsed=6.6min | ETA=223.9min
[DemoBuffer] Saved 18,840 transitions to saved/demos/expert_demos.npz
  Game   364/2000 | W=1.0% B=99.0% D=0.0% | trans=21,840 | elapsed=12.9min | ETA=211.2min
[DemoBuffer] Saved 21,840 transitions to saved/demos/expert_demos.npz
  Game   414/2000 | W=0.7% B=99.3% D=0.0% | trans=24,840 | elapsed=19.3min | ETA=204.0min
[DemoBuffer] Saved 24,840 transitions to saved/demos/expert_demos.npz

 Game  1798/2000 | W=0.0% B=100.0% D=0.0% | trans=107,880 | elapsed=6.3min | ETA=25.3min
[DemoBuffer] Saved 107,880 transitions to saved/demos/expert_demos.npz
  Game  1848/2000 | W=0.0% B=100.0% D=0.0% | trans=110,880 | elapsed=11.3min | ETA=17.2min
[DemoBuffer] Saved 110,880 transitions to saved/demos/expert_demos.npz
  Game  1898/2000 | W=0.0% B=100.0% D=0.0% | trans=113,880 | elapsed=17.6min | ETA=12.0min
[DemoBuffer] Saved 113,880 transitions to saved/demos/expert_demos.npz
  Game  1948/2000 | W=0.0% B=100.0% D=0.0% | trans=116,880 | elapsed=23.4min | ETA=6.1min
[DemoBuffer] Saved 116,880 transitions to saved/demos/expert_demos.npz
  Game  1998/2000 | W=0.0% B=100.0% D=0.0% | trans=119,880 | elapsed=28.6min | ETA=0.2min
[DemoBuffer] Saved 119,880 transitions to saved/demos/expert_demos.npz
[DemoBuffer] Saved 120,000 transitions to saved/demos/expert_demos.npz

[Generator] Complete — 2000 games in 28.9 minutes
[Generator] Total transitions: 120,000


[Pretrain] Using device: cuda
[DemoBuffer] Loaded 120,000 transitions from saved/demos/expert_demos.npz
[Pretrain] Loaded 120,000 expert transitions
[Pretrain] Starting fresh model
[Pretrain] Training 20 epochs, 120,000 transitions, batch=512

  Epoch  13/20 | td=0.0021 sup=0.0003 total=0.0024 | time=55.7s | elapsed=12.2min ETA=6.5min
[QNetwork] Saved to saved/models/pretrained_q.pt
           Best model saved (loss=0.0024)
  Epoch  14/20 | td=0.0020 sup=0.0003 total=0.0022 | time=56.2s | elapsed=13.1min ETA=5.6min
[QNetwork] Saved to saved/models/pretrained_q.pt
           Best model saved (loss=0.0022)
  Epoch  15/20 | td=0.0020 sup=0.0004 total=0.0024 | time=56.6s | elapsed=14.1min ETA=4.7min
  Epoch  16/20 | td=0.0019 sup=0.0003 total=0.0022 | time=56.0s | elapsed=15.0min ETA=3.7min
[QNetwork] Saved to saved/models/pretrained_q.pt
           Best model saved (loss=0.0022)
  Epoch  17/20 | td=0.0019 sup=0.0002 total=0.0020 | time=55.7s | elapsed=15.9min ETA=2.8min
[QNetwork] Saved to saved/models/pretrained_q.pt
           Best model saved (loss=0.0020)
  Epoch  18/20 | td=0.0019 sup=0.0002 total=0.0021 | time=55.6s | elapsed=16.9min ETA=1.9min
  Epoch  19/20 | td=0.0023 sup=0.0002 total=0.0024 | time=56.1s | elapsed=17.8min ETA=0.9min
  Epoch  20/20 | td=0.0048 sup=0.0001 total=0.0049 | time=56.2s | elapsed=18.7min ETA=0.0min

[Finetune] Epsilon: 0.3 → 0.05
[Finetune] Expert fraction in batches: 30%
------------------------------------------------------------
  Game    50/500 | ε=0.275 | W=6.0% L=92.0% D=2.0% | replay=123,000 | td=0.0156 sup=0.0001 | elapsed=0.9min ETA=8.5min
[QNetwork] Saved to saved/models/final_q.pt
  Game   100/500 | ε=0.251 | W=4.0% L=93.0% D=3.0% | replay=126,000 | td=0.0088 sup=0.0001 | elapsed=2.0min ETA=7.9min
  Game   150/500 | ε=0.225 | W=4.0% L=93.3% D=2.7% | replay=129,000 | td=0.0082 sup=0.0003 | elapsed=3.0min ETA=7.0min
  Game   200/500 | ε=0.200 | W=7.0% L=90.0% D=3.0% | replay=132,000 | td=0.0091 sup=0.0001 | elapsed=4.0min ETA=6.0min
[QNetwork] Saved to saved/models/final_q.pt
  Game   250/500 | ε=0.175 | W=7.6% L=90.0% D=2.4% | replay=135,000 | td=0.0101 sup=0.0003 | elapsed=5.0min ETA=5.0min
[QNetwork] Saved to saved/models/final_q.pt
  Game   300/500 | ε=0.150 | W=7.3% L=90.3% D=2.3% | replay=138,000 | td=0.0158 sup=0.0001 | elapsed=6.1min ETA=4.0min
  Game   350/500 | ε=0.126 | W=7.7% L=90.3% D=2.0% | replay=141,000 | td=0.0163 sup=0.0006 | elapsed=7.1min ETA=3.0min
[QNetwork] Saved to saved/models/final_q.pt
  Game   400/500 | ε=0.100 | W=7.8% L=90.2% D=2.0% | replay=144,000 | td=0.0126 sup=0.0002 | elapsed=8.2min ETA=2.0min
[QNetwork] Saved to saved/models/final_q.pt
  Game   450/500 | ε=0.075 | W=7.6% L=90.7% D=1.8% | replay=147,000 | td=0.0173 sup=0.0012 | elapsed=9.2min ETA=1.0min
  Game   500/500 | ε=0.050 | W=7.4% L=91.0% D=1.6% | replay=150,000 | td=0.0173 sup=0.0012 | elapsed=10.2min ETA=0.0min

[Finetune] Complete. Best model: saved/models/final_q.pt
[Finetune] Final win rate vs AlphaBeta-d1: 7.8%