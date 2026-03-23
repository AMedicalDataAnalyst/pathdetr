"""3-fold cross-validation runner for PanNuke.

Runs train_pannuke.py three times (one per fold rotation) and aggregates
test-fold results into a single summary.

Usage:
    python -m experiments.run_3fold_cv --data_root data --out_dir experiments/pannuke_3fold [... extra args]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

FOLD_ROTATIONS = [
    {"train_folds": [1, 2], "test_fold": 3},
    {"train_folds": [1, 3], "test_fold": 2},
    {"train_folds": [2, 3], "test_fold": 1},
]

SUMMARY_METRICS = ["mAP@30", "mAP@50", "PQ", "mPQ", "bPQ", "F1d@30", "mIoU"]


def parse_base_args() -> tuple[str, str, list[str]]:
    """Extract --data_root and --out_dir, forward everything else."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    known, remaining = parser.parse_known_args()
    return known.data_root, known.out_dir, remaining


def run_fold(data_root: str, out_dir: str, train_folds: list[int],
             test_fold: int, extra_args: list[str]) -> bool:
    """Run a single fold via subprocess. Returns True on success."""
    fold_dir = f"{out_dir}/fold{test_fold}"
    train_str = " ".join(str(f) for f in train_folds)
    cmd = [
        sys.executable, "-m", "experiments.train_pannuke",
        "--data_root", data_root,
        "--out_dir", fold_dir,
        "--train_folds", *[str(f) for f in train_folds],
        "--test_fold", str(test_fold),
        *extra_args,
    ]
    print(f"\n{'='*60}")
    print(f"FOLD: train {train_folds} / test {test_fold}")
    print(f"  out_dir: {fold_dir}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Fold {test_fold} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Fold {test_fold} failed: {e}")
        return False


def aggregate_results(out_dir: str) -> None:
    """Load per-fold test_results.json, print summary, save aggregated JSON."""
    fold_results = {}
    for rot in FOLD_ROTATIONS:
        tf = rot["test_fold"]
        path = Path(out_dir) / f"fold{tf}" / "test_results.json"
        if path.exists():
            with open(path) as f:
                fold_results[tf] = json.load(f)
        else:
            print(f"[WARN] Missing results for fold {tf}: {path}")

    if not fold_results:
        print("\nNo fold results found — nothing to aggregate.")
        return

    # Collect metric values across folds
    metric_values: dict[str, list[float]] = {m: [] for m in SUMMARY_METRICS}
    for tf, res in sorted(fold_results.items()):
        for m in SUMMARY_METRICS:
            if m in res:
                metric_values[m].append(res[m])

    # Print table
    header = f"{'Metric':<12}" + "".join(f"{'Fold'+str(t):>10}" for t in sorted(fold_results)) + f"{'Mean':>10}{'Std':>10}"
    print(f"\n{'='*len(header)}")
    print("3-Fold Cross-Validation Summary")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    summary: dict[str, dict] = {}
    for m in SUMMARY_METRICS:
        vals = metric_values[m]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        row = f"{m:<12}" + "".join(f"{v:>10.4f}" for v in vals) + f"{mean:>10.4f}{std:>10.4f}"
        print(row)
        summary[m] = {"per_fold": {str(t): fold_results[t].get(m) for t in sorted(fold_results)},
                       "mean": round(mean, 5), "std": round(std, 5)}

    print(f"{'='*len(header)}\n")

    # Save
    out_path = Path(out_dir) / "cv_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_path}")


def main() -> None:
    data_root, out_dir, extra_args = parse_base_args()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    successes = 0
    for rot in FOLD_ROTATIONS:
        ok = run_fold(data_root, out_dir, rot["train_folds"], rot["test_fold"], extra_args)
        if ok:
            successes += 1

    print(f"\n{successes}/{len(FOLD_ROTATIONS)} folds completed successfully.")
    aggregate_results(out_dir)


if __name__ == "__main__":
    main()
