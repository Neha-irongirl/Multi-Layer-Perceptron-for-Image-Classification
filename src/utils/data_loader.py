import numpy as np
from sklearn.model_selection import train_test_split

MNIST_CLASS_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _canonical_dataset_name(name):
    dataset_key = str(name).strip().lower()
    aliases = {
        "mnist": "mnist",
        "fashion_mnist": "fashion_mnist",
        "fashion-mnist": "fashion_mnist",
        "fashion mnist": "fashion_mnist",
    }
    if dataset_key not in aliases:
        raise ValueError("dataset must be one of: mnist, fashion_mnist")
    return aliases[dataset_key]


def _get_dataset_module(name):
    canonical = _canonical_dataset_name(name)

    try:
        if canonical == "mnist":
            from keras.datasets import mnist as ds
        else:
            from keras.datasets import fashion_mnist as ds
        return ds
    except Exception:
        try:
            if canonical == "mnist":
                from tensorflow.keras.datasets import mnist as ds
            else:
                from tensorflow.keras.datasets import fashion_mnist as ds
            return ds
        except Exception as exc:
            raise ImportError(
                "Unable to import dataset loader. Install TensorFlow-compatible Keras "
                "(for example: pip install tensorflow)."
            ) from exc


def get_class_names(name="mnist"):
    canonical = _canonical_dataset_name(name)
    if canonical == "mnist":
        return MNIST_CLASS_NAMES
    return FASHION_MNIST_CLASS_NAMES


def load_raw_dataset(name="mnist"):
    dataset_module = _get_dataset_module(name)
    (x_train, y_train), (x_test, y_test) = dataset_module.load_data()
    return {
        "x_train": x_train.astype(np.uint8),
        "y_train": y_train.astype(int),
        "x_test": x_test.astype(np.uint8),
        "y_test": y_test.astype(int),
        "class_names": get_class_names(name),
        "num_classes": 10,
    }


def load_dataset(name="mnist", val_size=0.1, random_state=42):
    dataset_module = _get_dataset_module(name)
    (x_train_full, y_train_full), (x_test, y_test) = dataset_module.load_data()

    x_train_full = x_train_full.astype(np.float64) / 255.0
    x_test = x_test.astype(np.float64) / 255.0

    x_train_full = x_train_full.reshape(x_train_full.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=float(val_size),
        random_state=int(random_state),
        stratify=y_train_full,
    )

    return {
        "x_train": x_train,
        "y_train": y_train.astype(int),
        "x_val": x_val,
        "y_val": y_val.astype(int),
        "x_test": x_test,
        "y_test": y_test.astype(int),
        "input_dim": x_train.shape[1],
        "num_classes": 10,
    }
