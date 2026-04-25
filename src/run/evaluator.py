# evaluator.py

import numpy as np


def inverse_scale(data, scaling, x_min, x_max):
    data = data / (scaling + 1e-8)
    return data * (x_max - x_min) + x_min


def evaluate(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))

    return rmse, mae, mape