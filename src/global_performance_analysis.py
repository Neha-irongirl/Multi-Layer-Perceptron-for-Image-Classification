from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def _latest_metric(run, key: str):
    try:
        value = _to_float(run.summary.get(key))
    except Exception:
        return None
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
    return float(values[-1])


def _max_metric(run, key: str):
    try:
        value = _to_float(run.summary.get(key))
    except Exception:
        return None
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
    return float(max(values))


def _get_runs(entity: str, project: str):
    api = wandb.Api()
    return api.runs(f"{entity}/{project}")


def _safe_attr(run: Any, key: str, default: str = "") -> str:
    try:
        value = getattr(run, key, default)
    except Exception:
        return default
    if value is None:
        return default
    return str(value)


def _safe_run_name(run: Any) -> str:
    name = _safe_attr(run, "name", "").strip()
    if name:
        return name
    run_id = _safe_attr(run, "id", "")
    return run_id if run_id else "unknown-run"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay Training vs Test Accuracy across all runs and flag overfitting gaps."
    )
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)

    parser.add_argument("--train_key", type=str, default="train/accuracy")
    parser.add_argument("--test_key", type=str, default="test/accuracy")
    parser.add_argument("--best_train_key", type=str, default="best_val_accuracy")
    parser.add_argument("--run_filter_state", type=str, default="finished", choices=["finished", "all"])
    parser.add_argument("--sort_by", type=str, default="created_at", choices=["created_at", "name", "id"])
    parser.add_argument("--sort_order", type=str, default="asc", choices=["asc", "desc"])

    parser.add_argument("--train_high", type=float, default=0.95)
    parser.add_argument("--gap_threshold", type=float, default=0.08)
    parser.add_argument("--max_runs", type=int, default=0, help="0 means all runs.")

    parser.add_argument("--out_plot", type=str, default="models/q2_7_train_vs_test_overlay.png")
    parser.add_argument("--out_csv", type=str, default="models/q2_7_overfitting_table.csv")

    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--log_project", type=str, default="global-performance-analysis")
    parser.add_argument("--log_run_name", type=str, default="q2_7-train-vs-test")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = list(_get_runs(args.entity, args.project))

    if args.run_filter_state != "all":
        runs = [r for r in runs if str(getattr(r, "state", "")).lower() == args.run_filter_state]

    if args.max_runs > 0:
        runs = runs[: args.max_runs]

    if not runs:
        print("No runs found with the current filters.")
        return

    reverse_sort = args.sort_order == "desc"
    if args.sort_by == "name":
        runs.sort(key=lambda r: _safe_run_name(r).lower(), reverse=reverse_sort)
    elif args.sort_by == "id":
        runs.sort(key=lambda r: _safe_attr(r, "id", ""), reverse=reverse_sort)
    else:
        runs.sort(key=lambda r: _safe_attr(r, "created_at", ""), reverse=reverse_sort)

    names = []
    train_accs = []
    test_accs = []
    best_val_accs = []
    skipped_runs = 0

    for run in runs:
        try:
            train_acc = _max_metric(run, args.train_key)
            test_acc = _latest_metric(run, args.test_key)
            best_val = _to_float(run.summary.get(args.best_train_key))
        except Exception:
            skipped_runs += 1
            continue

        if train_acc is None or test_acc is None:
            skipped_runs += 1
            continue

        names.append(_safe_run_name(run))
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        best_val_accs.append(best_val if best_val is not None else np.nan)

    if not names:
        print(
            "No valid runs with both train and test metrics found. "
            f"Checked train='{args.train_key}' and test='{args.test_key}'."
        )
        return

    train_arr = np.array(train_accs, dtype=np.float64)
    test_arr = np.array(test_accs, dtype=np.float64)
    best_val_arr = np.array(best_val_accs, dtype=np.float64)
    gap_arr = train_arr - test_arr
    overfit_mask = (train_arr >= args.train_high) & (gap_arr >= args.gap_threshold)
    overfit_idx = np.where(overfit_mask)[0]

    x = np.arange(len(names))
    fig = plt.figure(figsize=(14, 7))
    plt.plot(x, train_arr, marker="o", linewidth=1.5, label="Train Accuracy (max)")
    plt.plot(x, test_arr, marker="s", linewidth=1.5, label="Test Accuracy (final)")

    if np.isfinite(best_val_arr).any():
        plt.plot(x, best_val_arr, marker="^", linewidth=1.0, alpha=0.75, label="Best Val Accuracy")

    if overfit_idx.size > 0:
        plt.scatter(overfit_idx, train_arr[overfit_idx], color="red", s=60, label="Overfit candidates")
        plt.scatter(overfit_idx, test_arr[overfit_idx], color="red", s=60)

    plt.xticks(x, names, rotation=90)
    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.title("Global Performance: Train vs Test Accuracy Across Runs")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=220)
    print(f"Saved plot: {out_plot}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as fp:
        fp.write("run,train_accuracy_max,test_accuracy_final,best_val_accuracy,gap,flag_overfit\n")
        for i, name in enumerate(names):
            best_val = "" if np.isnan(best_val_arr[i]) else f"{best_val_arr[i]:.6f}"
            fp.write(
                f"{name},{train_arr[i]:.6f},{test_arr[i]:.6f},{best_val},{gap_arr[i]:.6f},{int(overfit_mask[i])}\n"
            )
    print(f"Saved table: {out_csv}")

    print("\nPotential overfitting runs (high train, lower test):")
    if overfit_idx.size == 0:
        print("- None detected under current thresholds.")
    else:
        for i in overfit_idx:
            print(
                f"- {names[i]} | train={train_arr[i]:.4f} | test={test_arr[i]:.4f} | gap={gap_arr[i]:.4f}"
            )

    interpretation = (
        "A large train-test gap indicates overfitting: the model learns training patterns well "
        "but generalizes poorly to unseen test data."
    )
    print("\nInterpretation:", interpretation)
    print(
        f"Included runs: {len(names)} | Skipped runs: {skipped_runs} | "
        f"Sort: {args.sort_by} ({args.sort_order})"
    )

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
                "best_train_key": args.best_train_key,
                "train_high": args.train_high,
                "gap_threshold": args.gap_threshold,
                "run_filter_state": args.run_filter_state,
                "sort_by": args.sort_by,
                "sort_order": args.sort_order,
                "included_runs": len(names),
                "skipped_runs": skipped_runs,
            },
        )
        wandb.log({"train_vs_test_overlay": wandb.Image(str(out_plot))})

        table = wandb.Table(
            columns=["run", "train_accuracy_max", "test_accuracy_final", "best_val_accuracy", "gap", "flag_overfit"]
        )
        for i, name in enumerate(names):
            table.add_data(
                name,
                float(train_arr[i]),
                float(test_arr[i]),
                None if np.isnan(best_val_arr[i]) else float(best_val_arr[i]),
                float(gap_arr[i]),
                bool(overfit_mask[i]),
            )
        wandb.log(
            {
                "run_table": table,
                "train_accuracy_line": wandb.plot.line(table, "run", "train_accuracy_max", title="Train Accuracy by Run"),
                "test_accuracy_line": wandb.plot.line(table, "run", "test_accuracy_final", title="Test Accuracy by Run"),
                "gap_line": wandb.plot.line(table, "run", "gap", title="Train-Test Gap by Run"),
            }
        )
        wandb.summary["num_runs"] = len(names)
        wandb.summary["num_overfit_runs"] = int(overfit_idx.size)
        wandb.summary["mean_gap"] = float(np.mean(gap_arr))
        wandb.summary["max_gap"] = float(np.max(gap_arr))
        wandb.summary["interpretation"] = interpretation
        wandb.save(str(out_plot))
        wandb.save(str(out_csv))
        wandb.finish()

    plt.close(fig)


if __name__ == "__main__":
    main()
