import wandb
from wandb.integration.keras import WandbCallback
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Initialize W&B
wandb.init(project="deep_learning",name='dataset_sample1')

# Load MNIST from Keras
(x_train, y_train), _ = keras.datasets.mnist.load_data()

# Normalize images (optional for visualization, but good practice)
x_train = x_train / 255.0

# Create dictionary to store 5 samples per class
class_samples = {i: [] for i in range(10)}

for img, label in zip(x_train, y_train):
    if len(class_samples[label]) < 5:
        class_samples[label].append(img)
    if all(len(v) == 5 for v in class_samples.values()):
        break

# Create W&B table
table = wandb.Table(columns=["class_label", "image"])

# Add images to table
for label in range(10):
    for img in class_samples[label]:
        table.add_data(
            label,
            wandb.Image(img, caption=f"Class {label}")
        )

# Log table
wandb.log({"MNIST Sample Images": table})

wandb.finish()

print("Logged 5 sample images per class to W&B!")