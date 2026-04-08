import csv
import os
from datetime import datetime


def save_evaluation_results(
    results:         dict,
    checkpoint_path: str,
    save_dir:        str = "data/training_logs",
):
    
    #Save evaluation results to a CSV file for tracking
    #agent strength across training iterations.
    
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "evaluation_results.csv")

    file_exists = os.path.exists(path)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'checkpoint', 'opponent',
            'wins', 'losses', 'draws', 'win_rate'
        ])
        if not file_exists:
            writer.writeheader()

        for opponent, stats in results.items():
            writer.writerow({
                'timestamp':  timestamp,
                'checkpoint': checkpoint_path,
                'opponent':   opponent,
                'wins':       stats['wins'],
                'losses':     stats['losses'],
                'draws':      stats['draws'],
                'win_rate':   f"{stats['win_rate']:.4f}",
            })

    print(f"[Metrics] Results saved to {path}")


def print_summary(results: dict):
    """Print a one-line summary of overall agent strength."""
    total_wins   = sum(s['wins']   for s in results.values())
    total_losses = sum(s['losses'] for s in results.values())
    total_draws  = sum(s['draws']  for s in results.values())
    total_games  = total_wins + total_losses + total_draws

    print(f"\n[Summary] Overall: {total_wins}W / {total_losses}L / {total_draws}D "
          f"across all opponents ({total_wins/total_games:.1%} win rate)")

    # Flag which opponents were beaten convincingly
    for name, stats in results.items():
        if stats['win_rate'] >= 0.55:
            print(f"  ✓ Beats {name} convincingly ({stats['win_rate']:.1%})")
        elif stats['win_rate'] >= 0.45:
            print(f"  ~ Even with {name} ({stats['win_rate']:.1%})")
        else:
            print(f"  ✗ Loses to {name} ({stats['win_rate']:.1%})")