"""
One-off, outside the repo, after 0180 and with no line-test job running or queued: move the best model's line-test
results from line_tests/ into line_tests/best/ (the variants' folders and the comparison stay where they are).
    python move_best_results.py [line_tests]            # lists the moves
    python move_best_results.py [line_tests] --apply
"""
import sys
from pathlib import Path

root = Path(next((a for a in sys.argv[1:] if a != "--apply"), "line_tests"))
best = root / "best"
moves = [p for p in sorted(root.iterdir()) if p.name not in {"best", "base", "best_poisson", "variant_comparison.csv"}]
for p in moves:
    print(f"{p} -> {best / p.name}")
if "--apply" in sys.argv:
    best.mkdir(exist_ok=True)
    for p in moves:
        p.rename(best / p.name)
print(f"{'Moved' if '--apply' in sys.argv else 'Would move'} {len(moves)} entries.")
