# Multi-Layer Perceptron for Image Classification

This project implements a modular MLP from scratch (NumPy-based math) for:
- MNIST
- Fashion-MNIST

## Install

```bash
pip install -r requirements.txt
```

## 1.1 Implementation Specifications (Assignment Mapping)

- Project structure:
  - `src/ann/activations.py`
  - `src/ann/neural_layers.py`
  - `src/ann/objective_functions.py`
  - `src/ann/optimizers.py`
  - `src/ann/neural_network.py`
  - `src/utils/data_loader.py`
  - `src/train.py`
  - `src/inference.py`
- Required CLI (train/inference) is implemented with `argparse`:
  - `-d/--dataset`, `-e/--epochs`, `-b/--batch_size`, `-l/--loss`,
    `-o/--optimizer`, `-lr/--learning_rate`, `-wd/--weight_decay`,
    `-nhl/--num_layers`, `-sz/--hidden_size`, `-a/--activation`,
    `-wi/--weight_init`.
- Gradient access:
  - Each dense layer exposes `grad_W` and `grad_b` after every backward pass.
- Inference metrics:
  - `src/inference.py` outputs Accuracy, Precision, Recall, and F1-score.
- Model artifacts:
  - Training saves `models/best_model.npy` and `models/best_config.json`.

## 1.2 Automated Evaluation Readiness

- Forward pass verification:
  - `NeuralNetwork.forward()` returns output logits (raw final-layer outputs).
- Gradient consistency:
  - Run numerical vs analytical check:

```bash
python src/gradient_check.py
```

- Training functionality:
  - `train.py` supports all required optimizer/loss/architecture settings and optional W&B logging.
- Private test performance artifact:
  - Save best weights to `models/best_model.npy` and evaluate with `src/inference.py`.
- Code quality:
  - Modular implementation and assignment-specific run commands are documented in this README.

## Clear MNIST Training Command

Use this command to train explicitly on **MNIST**:

```bash
python src/train.py --dataset mnist --epochs 10 --batch_size 32 --loss cross_entropy --optimizer adam --learning_rate 0.001 --weight_decay 0.0 --num_layers 3 --hidden_size 128,128,128 --activation relu --weight_init xavier
```

This saves:
- `models/best_model.npy`
- `models/best_config.json`

## Inference (MNIST)

```bash
python src/inference.py --dataset mnist --model_path models/best_model.npy
```

## Train on Fashion-MNIST

```bash
python src/train.py --dataset fashion_mnist --epochs 10 --batch_size 32 --loss cross_entropy --optimizer adam --learning_rate 0.001 --weight_decay 0.0 --num_layers 3 --hidden_size 128,128,128 --activation relu --weight_init xavier
```

## Step 2.1 (W&B Samples Table)

```bash
python notebooks/wandb_demo.py --dataset mnist --samples_per_class 5 --mode online
```

## Step 2.2 (W&B Sweep, 100 Runs)

```bash
python src/sweep.py --dataset mnist --count 100 --mode online --project da6401-assignment-1
```

## Step 2.3 (Optimizer Showdown)

```bash
python src/optimizer_showdown.py --dataset mnist --epochs 10 --batch_size 64 --learning_rate 0.001 --mode online
```

## Step 2.4 (Vanishing Gradient Analysis)

```bash
python src/vanishing_gradient_analysis.py --dataset mnist --epochs 10 --batch_size 64 --learning_rate 0.001 --num_layers_list 3,5 --mode online
```

## Step 2.5 (Dead Neuron Investigation)

```bash
python src/dead_neuron_investigation.py --dataset mnist --epochs 10 --learning_rate 0.1 --batch_size 64 --mode online
```

## Step 2.6 (Loss Comparison: MSE vs Cross-Entropy)

```bash
python src/loss_comparison.py --dataset mnist --epochs 10 --batch_size 64 --learning_rate 0.001 --mode online
```

## Step 2.7 (Global Performance Analysis)

```bash
python src/global_performance_analysis.py --entity <your_wandb_entity> --project da6401-assignment-1 --dataset mnist
```

## Step 2.8 (Error Analysis)

```bash
python src/error_analysis.py --dataset mnist --model_path models/best_model.npy --mode online
```

## Step 2.9 (Initialization Symmetry)

```bash
python src/initialization_symmetry.py --dataset mnist --iterations 50 --num_neurons 5 --layer_index 0 --mode online
```

## Step 2.10 (Fashion-MNIST Transfer Challenge)

```bash
python src/fashion_transfer_challenge.py --dataset fashion_mnist --epochs 12 --batch_size 64 --mode online
```

## W&B Report

- Project report: https://wandb.ai/ma24m018-iit-ma/deep_learning/reports/Multi-Layer-Perceptron-for-Image-Classification--VmlldzoxNjAyMzcyNw
