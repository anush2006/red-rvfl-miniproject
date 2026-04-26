import os
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import Ridge

from src.run.data_loader import load_dataset, create_dataset, split_data
from src.run.evaluator import inverse_scale, evaluate


DATASET_FOLDER = "./RVFL_Datasets"

datasets = [
    "DJI.xlsx", "HSI.xlsx", "KOSPI.xlsx", "LSE.xlsx",
    "NASDAQ.xlsx", "NIFTY50.xlsx", "NYSE.xlsx",
    "RUSSELL2000.xlsx", "SENSEX.xlsx", "SP500.xlsx", "SSE.xlsx"
]


WINDOW = 20
K = 1
SCALING = 0.7

# Only hyperparameter for Ridge
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10]


def scale_with_params(data, x_min, x_max, scaling):
    return ((data - x_min) / (x_max - x_min + 1e-8)) * scaling


results = []

for alpha in RIDGE_ALPHAS:

    print(f"\n[Ridge] Alpha = {alpha}")

    for file in datasets:

        path = os.path.join(DATASET_FOLDER, file)
        prices = load_dataset(path)

        X, y = create_dataset(prices.reshape(-1, 1), WINDOW, K)

        X = X.reshape(X.shape[0], -1)#time series take (batch, window, features) but ridge takes (batch, features)

        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

       

        x_min = X_train.min()
        x_max = X_train.max()

        X_train = scale_with_params(X_train, x_min, x_max, SCALING)
        X_val   = scale_with_params(X_val, x_min, x_max, SCALING)
        X_test  = scale_with_params(X_test, x_min, x_max, SCALING)

        y_train = scale_with_params(y_train, x_min, x_max, SCALING)
        y_val   = scale_with_params(y_val, x_min, x_max, SCALING)
        y_test  = scale_with_params(y_test, x_min, x_max, SCALING)

        # Train on train + val
        X_full = np.concatenate([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        start = time.time()

        model = Ridge(alpha=alpha)
        model.fit(X_full, y_full.ravel())

        training_time = time.time() - start

        pred = model.predict(X_test)

        y_test_orig = inverse_scale(y_test, SCALING, x_min, x_max).flatten()
        pred_orig   = inverse_scale(pred.reshape(-1,1), SCALING, x_min, x_max)

        rmse, mae, mape = evaluate(y_test_orig, pred_orig)

        print(f"{file} | RMSE: {rmse:.3f}")

        results.append([
            file, alpha,
            rmse, mae, mape, training_time
        ])


df = pd.DataFrame(results, columns=[
    "Dataset", "Alpha",
    "RMSE", "MAE", "MAPE", "Time"
])

df.to_csv("ridge_baseline_clean.csv", index=False)

print("\nClean Ridge baseline complete.")