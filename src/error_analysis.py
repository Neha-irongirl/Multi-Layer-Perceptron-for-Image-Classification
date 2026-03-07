from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import classification_report, confusion_matrix

from ann.neural_network import NeuralNetwork
from utils.data_loader import get_class_names, load_dataset, load_raw_dataset


def _to_float(value: Any):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _max_history_metric(run, key: str):
    try:
        v = _to_float(run.summary.get(key))
    except Exception:
        return None
    if v is not None:
        return v
    try:
        history = run.history(keys=[key], pandas=False)
    except Exception:
        return None
    values = []
    for row in history:
        x = _to_float(row.get(key))
        if x is not None:
            values.append(x)
    if not values:
        return None
    return float(max(values))


def _parse_neurons(value: Any, num_hidden_layers: int):
    if isinstance(value, int):
        return int(value)
    if isinstance(value, list):
        if len(value) == 1:
            return int(value[0])
        return [int(v) for v in value]
    if value is None:
        return 128
    text = str(value).strip()
    if "," not in text:
        return int(text)
    parts = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) == 1:
        return parts[0]
    return parts


def _choose_best_run(entity: str, project: str, metric_key: str):
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))
    if not runs:
        raise ValueError("No runs found in the given W&B project.")

    scored = []
    skipped = 0
    for run in runs:
        try:
            score = _max_history_metric(run, metric_key)
        except Exception:
            skipped += 1
            continue
        if score is not None:
            scored.append((score, run))
        else:
            skipped += 1
    if not scored:
        raise ValueError(f"No run contains metric '{metric_key}'.")
    if skipped > 0:
        print(f"Skipped {skipped} run(s) that were inaccessible or missing '{metric_key}'.")

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], float(scored[0][0])


def _build_model_from_run_cfg(cfg: dict[str, Any], input_dim: int):
    dataset = str(cfg.get("dataset", "mnist"))
    num_hidden_layers = int(cfg.get("num_hidden_layers", cfg.get("num_layers", 3)))
    num_neurons = _parse_neurons(cfg.get("num_neurons", cfg.get("hidden_size", 128)), num_hidden_layers)
    initialization_method = str(cfg.get("initialization_method", cfg.get("weight_init", "xavier")))
    activation_function = str(cfg.get("activation_function", cfg.get("activation", "relu")))
    optimizer = str(cfg.get("optimizer", "adam"))
    learning_rate = float(cfg.get("learning_rate", 1e-3))
    epochs = int(cfg.get("epochs", 10))
    batch_size = int(cfg.get("batch_size", 64))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    random_seed = int(cfg.get("random_seed", cfg.get("seed", 42)))

    model = NeuralNetwork(
        input_dim=input_dim,
        output_dim=10,
        num_neurons=num_neurons,
        num_hidden_layers=num_hidden_layers,
        initialization_method=initialization_method,
        activation_function=activation_function,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    resolved = {
        "dataset": dataset,
        "num_hidden_layers": num_hidden_layers,
        "num_neurons": num_neurons,
        "initialization_method": initialization_method,
        "activation_function": activation_function,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "random_seed": random_seed,
    }
    return model, resolved


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() * 0.5 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_top_confusion_pairs(cm: np.ndarray, class_names: list[str], out_path: Path, top_k: int = 12):
    off_diag = cm.copy().astype(np.int64)
    np.fill_diagonal(off_diag, 0)
    pairs = np.argwhere(off_diag > 0)

    if pairs.size == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No misclassifications detected.", ha="center", va="center")
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    pair_counts = []
    for i, j in pairs:
        pair_counts.append((int(off_diag[i, j]), int(i), int(j)))
    pair_counts.sort(reverse=True)
    top = pair_counts[:top_k]

    labels = [f"{class_names[i]} -> {class_names[j]}" for _, i, j in top]
    values = [c for c, _, _ in top]

    fig, ax = plt.subplots(figsize=(11, max(4, int(len(top) * 0.45))))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#d95f02")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of mistakes")
    ax.set_title("Creative Failure View: Most Frequent Confusion Pairs")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_misclassified_gallery(
    x_test_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    out_path: Path,
    max_items: int = 36,
):
    wrong_idx = np.where(y_true != y_pred)[0]
    if wrong_idx.size == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No misclassified samples.", ha="center", va="center")
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    pred_conf = y_prob[wrong_idx, y_pred[wrong_idx]]
    order = np.argsort(-pred_conf)
    chosen = wrong_idx[order[:max_items]]

    n = len(chosen)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.array(axes).reshape(rows, cols)

    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        ax.axis("off")
        if k >= n:
            continue

        idx = chosen[k]
        ax.imshow(x_test_raw[idx], cmap="gray")
        t = class_names[int(y_true[idx])]
        p = class_names[int(y_pred[idx])]
        conf = float(y_prob[idx, y_pred[idx]])
        ax.set_title(f"T:{t}\nP:{p} ({conf:.2f})", fontsize=8)

    plt.suptitle("Creative Failure View: Highest-Confidence Wrong Predictions", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Error analysis (confusion matrix + failure visualizations)."
    )
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--metric_key", type=str, default="best_val_accuracy")

    parser.add_argument("--val_size", type=float, default=None, help="Override validation split from run config.")
    parser.add_argument("--random_seed", type=int, default=None, help="Override random seed from run config.")
    parser.add_argument("--verbose_fit", action="store_true")

    parser.add_argument("--out_dir", type=str, default="models/error_analysis")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--log_project", type=str, default="error-analysis")
    parser.add_argument("--log_run_name", type=str, default="best-model-error-analysis")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    best_run, best_score = _choose_best_run(args.entity, args.project, args.metric_key)
    cfg = dict(best_run.config)

    dataset_name = str(cfg.get("dataset", "mnist"))
    val_size = float(args.val_size) if args.val_size is not None else float(cfg.get("val_ratio", 0.1))
    random_seed = int(args.random_seed) if args.random_seed is not None else int(cfg.get("random_seed", 42))

    data = load_dataset(dataset_name, val_size=val_size, random_state=random_seed)
    raw = load_raw_dataset(dataset_name)
    class_names = get_class_names(dataset_name)

    model, resolved_cfg = _build_model_from_run_cfg(cfg, input_dim=data["input_dim"])
    model.fit(
        data["x_train"],
        data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        verbose=args.verbose_fit,
    )

    y_true = data["y_test"].astype(int)
    y_prob = model.predict_proba(data["x_test"])
    y_pred = np.argmax(y_prob, axis=1).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    test_acc = float(np.mean(y_true == y_pred))
    errors = int(np.sum(y_true != y_pred))
    error_rate = float(errors / y_true.size)

    out_dir = Path(args.out_dir)
    cm_path = out_dir / "confusion_matrix.png"
    pairs_path = out_dir / "top_confusion_pairs.png"
    gallery_path = out_dir / "misclassified_gallery.png"
    report_path = out_dir / "classification_report.txt"

    _plot_confusion_matrix(cm, class_names, cm_path)
    _plot_top_confusion_pairs(cm, class_names, pairs_path)
    _plot_misclassified_gallery(raw["x_test"], y_true, y_pred, y_prob, class_names, gallery_path)

    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Selected best run: {best_run.name or best_run.id}")
    print(f"Selection metric {args.metric_key}: {best_score:.4f}")
    print(f"Re-trained test accuracy: {test_acc:.4f}")
    print(f"Test errors: {errors}/{y_true.size} (error_rate={error_rate:.4f})")
    print("Confusion matrix class order:", class_names)
    print("Confusion matrix:")
    print(cm)
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved top confusion pairs: {pairs_path}")
    print(f"Saved misclassified gallery: {gallery_path}")
    print(f"Saved classification report: {report_path}")

    if args.log_to_wandb:
        wandb.init(
            project=args.log_project,
            entity=args.entity,
            mode=args.wandb_mode,
            name=args.log_run_name,
            config={
                "source_project": args.project,
                "source_run_id": best_run.id,
                "source_metric_key": args.metric_key,
                "source_metric_value": best_score,
                **resolved_cfg,
                "val_size": val_size,
            },
        )
        cm_table = wandb.Table(columns=["true_label", "pred_label", "count"])
        for i, true_name in enumerate(class_names):
            for j, pred_name in enumerate(class_names):
                cm_table.add_data(true_name, pred_name, int(cm[i, j]))

        wandb.log(
            {
                "confusion_matrix": wandb.Image(str(cm_path)),
                "confusion_matrix_raw": cm_table,
                "top_confusion_pairs": wandb.Image(str(pairs_path)),
                "misclassified_gallery": wandb.Image(str(gallery_path)),
                "test_accuracy": test_acc,
                "test_error_rate": error_rate,
                "test_errors": errors,
            }
        )
        wandb.summary["classification_report"] = report_text
        wandb.finish()


if __name__ == "__main__":
    main()
