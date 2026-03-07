import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def _load_default_config():
    config_path = Path(__file__).resolve().parent / "config" / "hyperparameters.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_parser():
    defaults = _load_default_config()
    parser = argparse.ArgumentParser(description="Inference for saved MLP model")
    parser.add_argument("-d", "--dataset", type=str, default=defaults.get("dataset", "mnist"), choices=["mnist", "fashion_mnist"])
    # Compatibility args required in assignment CLI contract.
    parser.add_argument("-e", "--epochs", type=int, default=defaults.get("epochs", 10))
    parser.add_argument("-b", "--batch_size", type=int, default=defaults.get("batch_size", 32))
    parser.add_argument("-l", "--loss", type=str, default=defaults.get("loss", "cross_entropy"), choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", type=str, default=defaults.get("optimizer", "adam"), choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=defaults.get("learning_rate", 1e-3))
    parser.add_argument("-wd", "--weight_decay", type=float, default=defaults.get("weight_decay", 0.0))
    parser.add_argument("-nhl", "--num_layers", type=int, default=defaults.get("num_layers", 3))
    parser.add_argument("-sz", "--hidden_size", nargs="+", default=defaults.get("hidden_size", ["128", "128", "128"]))
    parser.add_argument("-a", "--activation", type=str, default=defaults.get("activation", "relu"), choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-wi", "--weight_init", type=str, default=defaults.get("weight_init", "xavier"), choices=["random", "xavier", "zeros"])
    parser.add_argument("--model_path", type=str, default=defaults.get("model_path", "models/best_model.npy"))
    return parser


def main():
    args = build_parser().parse_args()

    defaults = _load_default_config()
    data = load_dataset(
        args.dataset,
        val_size=float(defaults.get("val_size", 0.1)),
        random_state=int(defaults.get("seed", 42)),
    )
    model = NeuralNetwork.load_from_file(args.model_path)

    y_true = data["y_test"]
    y_pred = model.predict(data["x_test"])

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Dataset:   {args.dataset}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


if __name__ == "__main__":
    main()
