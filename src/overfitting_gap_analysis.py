from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _metric_from_run(run, key: str):
    value = _to_float(run.summary.get(key))
    if value is not None:
        return value

    try:
        history = run.history(keys=[key], pandas=False)
    except Exception:
        return None

    values = []
    for row in history:
        v = _to_float(row.get(key))
        if v is not None:
            values.append(v)

    if not values:
        return None
    return max(values)


def _get_runs(entity: str, project: str):
    api = wandb.Api()
    try:
        return api.runs(f"{entity}/{project}")
    except Exception as exc:
        print(f"Could not access project '{project}' under entity '{entity}': {exc}")
        try:
            projects = [p.name for p in api.projects(entity)]
            if projects:
                print("Available projects for this entity:")
                for p in projects:
                    print(f"- {p}")
            else:
                print("No visible projects found for this entity.")
        except Exception:
            print("Could not fetch project list from W&B API.")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay Training vs Test Accuracy for W&B hyperparameter runs"
    )

    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True, help="Source project to read runs from")

    parser.add_argument("--train_key", type=str, default="train/accuracy")
    parser.add_argument("--test_key", type=str, default="test/accuracy")
    parser.add_argument("--train_high", type=float, default=0.95)
    parser.add_argument("--gap_threshold", type=float, default=0.08)
    parser.add_argument("--out", type=str, default="models/train_vs_test_overlay.png")

    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--log_project", type=str, default="overfitting-analysis")
    parser.add_argument("--log_run_name", type=str, default="train-vs-test-overlay")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    args = parser.parse_args()

    runs = _get_runs(args.entity, args.project)

    names = []
    train_accs = []
    test_accs = []

    for run in runs:
        train_acc = _metric_from_run(run, args.train_key)
        test_acc = _metric_from_run(run, args.test_key)
        if train_acc is None or test_acc is None:
            continue
        names.append(run.name or run.id)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    if not names:
        print(
            "No runs with both metrics found. "
            f"Checked keys train='{args.train_key}', test='{args.test_key}'."
        )
        return

    train_arr = np.array(train_accs, dtype=np.float64)
    test_arr = np.array(test_accs, dtype=np.float64)
    gap = train_arr - test_arr

    overfit_idx = np.where((train_arr >= args.train_high) & (gap >= args.gap_threshold))[0]

    x = np.arange(len(names))
    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, train_arr, marker="o", label="Training Accuracy")
    plt.plot(x, test_arr, marker="s", label="Test Accuracy")

    if overfit_idx.size > 0:
        plt.scatter(overfit_idx, train_arr[overfit_idx], s=70, color="red", label="Overfit runs")
        plt.scatter(overfit_idx, test_arr[overfit_idx], s=70, color="red")

    plt.xticks(x, names, rotation=90)
    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy Across Hyperparameter Runs")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved overlay plot: {out_path}")

    if overfit_idx.size == 0:
        print("No strong overfitting runs found with current thresholds.")
    else:
        print("Potential overfitting runs:")
        for idx in overfit_idx:
            print(
                f"- {names[idx]} | train_acc={train_arr[idx]:.4f} | "
                f"test_acc={test_arr[idx]:.4f} | gap={gap[idx]:.4f}"
            )

    interpretation = (
        "High training accuracy with low test accuracy indicates overfitting: "
        "the model memorizes training data but generalizes poorly to unseen data."
    )
    print("\nInterpretation:", interpretation)

    if args.log_to_wandb:
        wandb.init(
            project=args.log_project,
            entity=args.entity,
            mode=args.wandb_mode,
            name=args.log_run_name,
            config={
                "source_project": args.project,
                "train_key": args.train_key,
                "test_key": args.test_key,
                "train_high": args.train_high,
                "gap_threshold": args.gap_threshold,
            },
        )

        wandb.log({"overlay/train_vs_test": wandb.Image(fig)})

        table = wandb.Table(columns=["run", "train_accuracy", "test_accuracy", "gap", "flag_overfit"])
        for i, name in enumerate(names):
            is_overfit = bool((train_arr[i] >= args.train_high) and (gap[i] >= args.gap_threshold))
            table.add_data(name, float(train_arr[i]), float(test_arr[i]), float(gap[i]), is_overfit)
        wandb.log({"overfitting/run_table": table})

        wandb.summary["num_runs"] = len(names)
        wandb.summary["num_overfit_runs"] = int(overfit_idx.size)
        wandb.summary["mean_gap"] = float(np.mean(gap))
        wandb.summary["max_gap"] = float(np.max(gap))
        wandb.summary["interpretation"] = interpretation

        wandb.finish()

    plt.close(fig)


if __name__ == "__main__":
    main()
