# data_loader.py

import pandas as pd
import numpy as np


def load_dataset(path):
    df = pd.read_excel(path, skiprows=2)
    df.columns = ['Date', 'Close']
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    return df['Close'].values


def scale_data(data, scaling):
    x_min = data.min()
    x_max = data.max()

    scaled = (data - x_min) / (x_max - x_min + 1e-8)
    scaled = scaled * scaling

    return scaled.reshape(-1, 1), (x_min, x_max)


def create_dataset(data, window, k):
    X, y = [], []

    for i in range(window + k - 1, len(data)):
        features = []

        for j in range(k):
            start = i - window - j
            end   = i - j
            features.append(data[start:end])

        X.append(np.concatenate(features))
        y.append(data[i])

    return np.array(X), np.array(y)


def split_data(X, y):
    n = len(X)

    train_end = int(n * 0.7)
    val_end   = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test