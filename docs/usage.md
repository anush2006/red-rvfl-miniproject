# RedRVFL Architecture Usage Guide

## Overview

This module implements the **feature extraction architecture** used in the
Recurrent Ensemble Deep Random Vector Functional Link (RedRVFL) model.

The module provides:

- Random LSTM feature extraction
- Deep layer input propagation
- RVFL feature matrix construction

The module **does not perform training**.  
Training (ridge regression, hyperparameter tuning, metrics) is handled
externally by the training pipeline.

---

# Architecture Summary

The RedRVFL architecture combines:

1. **Randomized recurrent feature extraction**
2. **RVFL feature construction**
3. **Layer-wise ensemble predictions**

Key property:

- LSTM weights are **randomly initialized and frozen**
- Only the **ridge regression output layer is trained**

---

# Data Flow

## 1. Input Window

The model expects a PyTorch tensor:


(batch_size, window_size, input_features)


For the financial time series experiments:


input_features = 1 (closing price)


Example:


(128, 96, 1)


---

## 2. Random LSTM Feature Extraction

The input sequence is passed through a **Random LSTM**:


sequence → LSTM → hidden representation


Output shape:


(batch_size, hidden_size)


Example:


(128, 64)


This hidden representation captures temporal patterns in the input window.

---

## 3. Feature Matrix Construction (RVFL)

The RVFL architecture concatenates:


D = [hidden_features , flattened_input]


Where:


flattened_input shape
(batch_size, window_size * input_features)


Example:


hidden_size = 64
window_size = 96
input_features = 1

flattened_input = 96
hidden_features = 64

D = 160 features


Final matrix shape:


(batch_size, hidden_size + window_size * input_features)

Example:
(128, 160)


This matrix **D** is passed to ridge regression for training.

---

# Deep RedRVFL Layer Propagation

RedRVFL supports multiple layers.

### Layer 1

Input:


X


### Layer l (l > 1)

Input:


[h_(l-1), X]


Where:


h_(l-1) = hidden representation from previous layer


This allows deeper layers to use both:

- original time series input
- previous layer features

---

# Ensemble Prediction

Each layer produces a prediction:


ŷ₁
ŷ₂
...
ŷ_L


The final prediction is computed using:


median(ŷ₁, ŷ₂, ..., ŷ_L)


Median aggregation improves robustness and reduces sensitivity to outliers.

---

# Module API

The architecture module provides the following components.

---

## RandomLSTM

Random LSTM feature extractor.

### Initialization

```python
lstm = RandomLSTM(input_size, hidden_size)
Input
(batch_size, window_size, features)
Output
(batch_size, hidden_size)

Example:

hidden = lstm(X_tensor)
flatten_window

Flattens the input time window.

flattened = flatten_window(X_tensor)

Output shape:

(batch_size, window_size * features)
build_layer_input

Constructs the input for deeper RedRVFL layers.

[h_(l-1), X]

Example:

layer_input = build_layer_input(previous_hidden, xtensor)

Output shape:

(batch_size, window_size, features + hidden_size)
build_feature_matrix

Constructs the RVFL feature matrix used by ridge regression.

D = [hidden_features , flattened_input]

Example:

D = build_feature_matrix(xtensor, hidden_tensor)

Output:

numpy array
(batch_size, hidden_size + window_size * features)
Example Usage

Example feature extraction pipeline:

from architecture import RandomLSTM
from architecture import build_layer_input
from architecture import build_feature_matrix

lstm = RandomLSTM(input_size=1, hidden_size=64)

hidden = lstm(X_tensor)

D = build_feature_matrix(X_tensor, hidden)

For deeper layers:

layer_input = build_layer_input(hidden, X_tensor)

hidden_2 = lstm_layer2(layer_input)
Tensor / NumPy Conventions

The architecture uses both PyTorch and NumPy.

Feature extraction → PyTorch tensors
Feature matrix D → NumPy arrays

This allows compatibility with scikit-learn ridge regression.


