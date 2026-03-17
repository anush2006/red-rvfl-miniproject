import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def inverse_transform(y, x_min, x_max):
    return y * (x_max - x_min) + x_min

def rmse(y_true, y_pred, x_min=None, x_max=None):
    if x_min is not None and x_max is not None:
        y_true = inverse_transform(y_true, x_min, x_max)
        y_pred = inverse_transform(y_pred, x_min, x_max)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred, x_min=None, x_max=None):
    if x_min is not None and x_max is not None:
        y_true = inverse_transform(y_true, x_min, x_max)
        y_pred = inverse_transform(y_pred, x_min, x_max)
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred, x_min=None, x_max=None):
    if x_min is not None and x_max is not None:
        y_true = inverse_transform(y_true, x_min, x_max)
        y_pred = inverse_transform(y_pred, x_min, x_max)
    return np.mean(np.abs((y_true - y_pred) / y_true))