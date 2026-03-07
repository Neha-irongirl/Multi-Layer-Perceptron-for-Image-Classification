import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.data_loader import get_class_names, load_raw_dataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="Step 2.1: Log 5 sample images per class to W&B."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
    )
    parser.add_argument("--samples_per_class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="da6401-assignment-1")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    return parser


def select_samples_per_class(labels, samples_per_class, seed):
    rng = np.random.default_rng(int(seed))
    class_ids = sorted(np.unique(labels).tolist())

    selected = []
    for class_id in class_ids:
        class_indices = np.where(labels == class_id)[0]
        if class_indices.shape[0] < int(samples_per_class):
            raise ValueError(
                f"Class {class_id} has only {class_indices.shape[0]} samples, "
                f"cannot pick {samples_per_class}."
            )
        chosen = rng.choice(
            class_indices,
            size=int(samples_per_class),
            replace=False,
        )
        for idx in sorted(chosen.tolist()):
            selected.append((int(class_id), int(idx)))
    return selected


def get_similarity_analysis(dataset):
    if dataset == "mnist":
        similar_pairs = [
            ("4", "9", "Similar loop and vertical stroke structure."),
            ("3", "5", "Curved strokes can overlap in raw handwritten forms."),
            ("1", "7", "Slanted writing styles make these look close."),
        ]
    else:
        similar_pairs = [
            ("T-shirt/top", "Shirt", "Both are upper-wear silhouettes at low resolution."),
            ("Pullover", "Coat", "Long-sleeve shapes are visually similar in 28x28 pixels."),
            ("Sandal", "Sneaker", "Footwear outlines overlap in grayscale thumbnails."),
        ]
    impact_note = (
        "Visual similarity increases inter-class overlap in feature space. "
        "This typically raises confusion between similar classes and lowers "
        "precision/recall for those categories."
    )
    return similar_pairs, impact_note


def main():
    args = build_parser().parse_args()
    if args.mode == "disabled":
        print("W&B mode is disabled. Nothing to log.")
        return

    os.environ["WANDB_SILENT"] = "true"
    import wandb

    data = load_raw_dataset(args.dataset)
    class_names = get_class_names(args.dataset)

    selected = select_samples_per_class(
        data["y_train"],
        args.samples_per_class,
        args.seed,
    )

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        mode=args.mode,
        config={
            "dataset": args.dataset,
            "samples_per_class": args.samples_per_class,
            "seed": args.seed,
        },
    )

    table = wandb.Table(columns=["class_id", "class_name", "image"])
    for class_id, idx in selected:
        table.add_data(
            class_id,
            class_names[class_id],
            wandb.Image(data["x_train"][idx], caption=f"{class_names[class_id]} ({class_id})"),
        )

    run.log({"data_exploration_samples": table})

    similar_pairs, impact_note = get_similarity_analysis(args.dataset)

    similarity_table = wandb.Table(
        columns=["class_a", "class_b", "visual_similarity_note", "expected_model_impact"]
    )
    for class_a, class_b, note in similar_pairs:
        similarity_table.add_data(
            class_a,
            class_b,
            note,
            "Likely higher confusion for this pair, reducing per-class precision/recall.",
        )

    run.log(
        {
            "data_exploration_similarity_table": similarity_table,
            "data_exploration_impact_note": wandb.Html(f"<p>{impact_note}</p>"),
        }
    )
    run.finish()
    print("Logged W&B table: data_exploration_samples")
    print("Logged similarity analysis: data_exploration_similarity_table")


if __name__ == "__main__":
    main()
