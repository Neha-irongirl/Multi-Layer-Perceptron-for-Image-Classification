"""Fully connected neural network implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .activations import get_activation
from .objective_functions import accuracy_from_probs, cross_entropy_loss, one_hot, softmax
from .optimizers import Optimizer, get_optimizer


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
class NeuralNetwork:
    """Configurable MLP for classification."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_neurons: int | Iterable[int] = 128,
        num_hidden_layers: int = 2,
        initialization_method: str = "xavier",
        activation_function: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        weight_decay: float = 0.0,
        random_seed: int | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.initialization_method = initialization_method.lower()
        self.activation_function = activation_function.lower()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.random_seed = random_seed

        self.hidden_layer_sizes = self._resolve_hidden_sizes(num_neurons, self.num_hidden_layers)
        self.layer_dims = [self.input_dim, *self.hidden_layer_sizes, self.output_dim]
        self.num_layers = len(self.layer_dims) - 1

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self.activation, self.activation_derivative = get_activation(self.activation_function)
        self.params = self._initialize_parameters()
        self.optimizer: Optimizer = get_optimizer(optimizer, self.learning_rate)

    @staticmethod
    def _resolve_hidden_sizes(num_neurons: int | Iterable[int], num_hidden_layers: int) -> list[int]:
        if isinstance(num_neurons, int):
            return [int(num_neurons)] * num_hidden_layers

        sizes = [int(x) for x in num_neurons]
        if len(sizes) != num_hidden_layers:
            raise ValueError(
                "Length of num_neurons list must match num_hidden_layers. "
                f"Got {len(sizes)} and {num_hidden_layers}."
            )
        return sizes

    def _initialize_parameters(self) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {}

        for layer in range(1, self.num_layers + 1):
            fan_in = self.layer_dims[layer - 1]
            fan_out = self.layer_dims[layer]
            params[f"b{layer}"] = np.zeros((1, fan_out), dtype=np.float64)

            if self.initialization_method == "xavier":
                std = np.sqrt(2.0 / (fan_in + fan_out))
                params[f"W{layer}"] = np.random.randn(fan_in, fan_out) * std
            elif self.initialization_method == "he":
                std = np.sqrt(2.0 / fan_in)
                params[f"W{layer}"] = np.random.randn(fan_in, fan_out) * std
            elif self.initialization_method == "normal":
                params[f"W{layer}"] = np.random.randn(fan_in, fan_out) * 0.01
            elif self.initialization_method == "zeros":
                params[f"W{layer}"] = np.zeros((fan_in, fan_out), dtype=np.float64)
            else:
                supported = "xavier, he, normal, zeros"
                raise ValueError(
                    f"Unsupported initialization method '{self.initialization_method}'. Supported: {supported}"
                )

            params[f"W{layer}"] = params[f"W{layer}"].astype(np.float64, copy=False)

        return params

    def _forward(self, x: np.ndarray):
        caches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        activation_prev = x

        for layer in range(1, self.num_layers):
            weights = self.params[f"W{layer}"]
            bias = self.params[f"b{layer}"]
            z = activation_prev @ weights + bias
            activation_next = self.activation(z)
            caches.append((activation_prev, z, weights, bias))
            activation_prev = activation_next

        weights_out = self.params[f"W{self.num_layers}"]
        bias_out = self.params[f"b{self.num_layers}"]
        logits = activation_prev @ weights_out + bias_out
        probs = softmax(logits)
        caches.append((activation_prev, logits, weights_out, bias_out))
        return probs, caches

    def _compute_loss(self, probs: np.ndarray, targets: np.ndarray) -> float:
        base_loss = cross_entropy_loss(probs, targets)
        if self.weight_decay <= 0.0:
            return base_loss

        l2 = 0.0
        for layer in range(1, self.num_layers + 1):
            weights = self.params[f"W{layer}"]
            l2 += np.sum(weights * weights)
        return float(base_loss + 0.5 * self.weight_decay * l2 / targets.shape[0])

    def _backward(
        self, probs: np.ndarray, targets: np.ndarray, caches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        grads: dict[str, np.ndarray] = {}
        batch_size = targets.shape[0]

        dlogits = (probs - targets) / batch_size
        prev_activation, _, weights, _ = caches[-1]
        grads[f"W{self.num_layers}"] = prev_activation.T @ dlogits + (self.weight_decay / batch_size) * weights
        grads[f"b{self.num_layers}"] = np.sum(dlogits, axis=0, keepdims=True)

        dactivation = dlogits @ weights.T

        for layer in range(self.num_layers - 1, 0, -1):
            prev_activation, z, weights, _ = caches[layer - 1]
            dz = dactivation * self.activation_derivative(z)
            grads[f"W{layer}"] = prev_activation.T @ dz + (self.weight_decay / batch_size) * weights
            grads[f"b{layer}"] = np.sum(dz, axis=0, keepdims=True)
            dactivation = dz @ weights.T

        return grads

    @staticmethod
    def _iter_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int):
        num_samples = x.shape[0]
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield x[batch_idx], y[batch_idx]

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        x_train = x_train.astype(np.float64)
        y_train = y_train.astype(int).ravel()
        history = TrainingHistory()
        train_targets = one_hot(y_train, self.output_dim)

        for epoch in range(1, self.epochs + 1):
            for xb, yb in self._iter_minibatches(x_train, y_train, self.batch_size):
                yb_onehot = one_hot(yb, self.output_dim)
                probs, caches = self._forward(xb)
                grads = self._backward(probs, yb_onehot, caches)
                self.optimizer.step(self.params, grads)

            train_probs, _ = self._forward(x_train)
            train_loss = self._compute_loss(train_probs, train_targets)
            train_acc = accuracy_from_probs(train_probs, y_train)
            history.train_loss.append(train_loss)
            history.train_accuracy.append(train_acc)

            if x_val is not None and y_val is not None:
                y_val = y_val.astype(int).ravel()
                val_probs, _ = self._forward(x_val.astype(np.float64))
                val_targets = one_hot(y_val, self.output_dim)
                val_loss = self._compute_loss(val_probs, val_targets)
                val_acc = accuracy_from_probs(val_probs, y_val)
                history.val_loss.append(val_loss)
                history.val_accuracy.append(val_acc)

            if verbose:
                if history.val_loss:
                    print(
                        f"Epoch {epoch:03d}/{self.epochs} | "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                        f"val_loss={history.val_loss[-1]:.4f} val_acc={history.val_accuracy[-1]:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:03d}/{self.epochs} | "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
                    )

        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs, _ = self._forward(x.astype(np.float64))
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        y = y.astype(int).ravel()
        targets = one_hot(y, self.output_dim)
        probs = self.predict_proba(x)
        loss = self._compute_loss(probs, targets)
        acc = accuracy_from_probs(probs, y)
        return loss, acc
