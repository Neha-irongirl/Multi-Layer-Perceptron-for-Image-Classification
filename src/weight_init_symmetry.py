from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from ann.objective_functions import one_hot
from utils.data_loader import load_dataset


def run_and_track(
    x_train: np.ndarray,
    y_train: np.ndarray,
    init_method: str,
    hidden_layers: int,
    hidden_size: int,
    activation: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    random_seed: int,
    iterations: int,
    track_layer: int,
    track_neurons: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = NeuralNetwork(
        input_dim=x_train.shape[1],
        output_dim=10,
        num_neurons=hidden_size,
        num_hidden_layers=hidden_layers,
        initialization_method=init_method,
        activation_function=activation,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=1,
        batch_size=batch_size,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )

    grad_hist = np.zeros((iterations, track_neurons), dtype=np.float64)
    loss_hist = np.zeros(iterations, dtype=np.float64)

    step = 0
    while step < iterations:
        for xb, yb in model._iter_minibatches(x_train, y_train, model.batch_size):
            probs, caches = model._forward(xb)
            targets = one_hot(yb, model.output_dim)
            loss_hist[step] = model._compute_loss(probs, targets)

            grads = model._backward(probs, targets, caches)
            dW = grads[f"W{track_layer}"]
            cols = min(track_neurons, dW.shape[1])
            grad_hist[step, :cols] = np.linalg.norm(dW[:, :cols], axis=0)

            model.optimizer.step(model.params, grads)

            step += 1
            if step >= iterations:
                break

    return grad_hist, loss_hist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare zeros vs xavier initialization and track per-neuron gradients."
    )
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--track_layer", type=int, default=1, help="1-based hidden/output layer index for W gradients.")
    parser.add_argument("--track_neurons", type=int, default=5)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--wandb_project", type=str, default="weight-init-symmetry")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_run_name", type=str, default="zeros_vs_xavier")
    parser.add_argument("--out_plot", type=str, default="models/gradient_symmetry.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    np.random.seed(args.random_seed)
    data = load_dataset(args.dataset, val_size=args.val_size, random_state=args.random_seed)
    x_train, y_train = data["x_train"], data["y_train"]

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.wandb_run_name,
        config=vars(args),
    )

    zeros_grad, zeros_loss = run_and_track(
        x_train=x_train,
        y_train=y_train,
        init_method="zeros",
        hidden_layers=args.hidden_layers,
        hidden_size=args.hidden_size,
        activation=args.activation,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        random_seed=args.random_seed,
        iterations=args.iterations,
        track_layer=args.track_layer,
        track_neurons=args.track_neurons,
    )

    xavier_grad, xavier_loss = run_and_track(
        x_train=x_train,
        y_train=y_train,
        init_method="xavier",
        hidden_layers=args.hidden_layers,
        hidden_size=args.hidden_size,
        activation=args.activation,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        random_seed=args.random_seed,
        iterations=args.iterations,
        track_layer=args.track_layer,
        track_neurons=args.track_neurons,
    )

    iters = np.arange(1, args.iterations + 1)
    for i in range(args.iterations):
        payload = {
            "iteration": int(iters[i]),
            "zeros/loss": float(zeros_loss[i]),
            "xavier/loss": float(xavier_loss[i]),
        }
        for n in range(args.track_neurons):
            payload[f"zeros/grad_neuron_{n+1}"] = float(zeros_grad[i, n])
            payload[f"xavier/grad_neuron_{n+1}"] = float(xavier_grad[i, n])
        wandb.log(payload, step=int(iters[i]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for n in range(args.track_neurons):
        axes[0].plot(iters, zeros_grad[:, n], label=f"Neuron {n+1}")
        axes[1].plot(iters, xavier_grad[:, n], label=f"Neuron {n+1}")

    axes[0].set_title("Zeros Initialization")
    axes[1].set_title("Xavier Initialization")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"||dW(layer {args.track_layer})[:, neuron]||_2")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(" Gradient Trajectories of 5 Neurons (First 50 Iterations)")
    fig.tight_layout()

    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=220)
    wandb.log({"gradient_plot": wandb.Image(fig)})
    plt.close(fig)

    loss_fig, loss_ax = plt.subplots(figsize=(8, 5))
    loss_ax.plot(iters, zeros_loss, label="Zeros loss", linewidth=2)
    loss_ax.plot(iters, xavier_loss, label="Xavier loss", linewidth=2)
    loss_ax.set_xlabel("Iteration")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Loss Comparison (First 50 Iterations)")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend(loc="best")
    loss_fig.tight_layout()

    loss_plot = out_plot.parent / "loss_comparison.png"
    loss_fig.savefig(loss_plot, dpi=220)
    wandb.log({"loss_plot": wandb.Image(loss_fig)})
    plt.close(loss_fig)

    zeros_std = np.mean(np.std(zeros_grad, axis=1))
    xavier_std = np.mean(np.std(xavier_grad, axis=1))
    wandb.summary["zeros/mean_neuron_gradient_std"] = float(zeros_std)
    wandb.summary["xavier/mean_neuron_gradient_std"] = float(xavier_std)
    wandb.summary["zeros/loss_start"] = float(zeros_loss[0])
    wandb.summary["zeros/loss_end"] = float(zeros_loss[-1])
    wandb.summary["xavier/loss_start"] = float(xavier_loss[0])
    wandb.summary["xavier/loss_end"] = float(xavier_loss[-1])
    wandb.finish()

    print(f"Saved gradient plot: {out_plot}")
    print(f"Saved loss plot: {loss_plot}")


if __name__ == "__main__":
    main()
