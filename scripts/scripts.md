# Phase 1 — generate expert demos (resume-safe)
python scripts/generate_demos.py

# Phase 2 — pre-train Q-network (resume-safe)
python scripts/pretrain.py

# Phase 3 — fine-tune via self-play (resume-safe)
python scripts/finetune.py

# Competition
python scripts/compete.py