import os
import time
import pandas as pd
import numpy as np

from src.run.config import get_all_configs
from src.run.data_loader import load_dataset, create_dataset, split_data
from src.run.model_runner import train_model, predict
from src.run.evaluator import inverse_scale, evaluate


DATASET_FOLDER = "./RVFL_Datasets"

datasets = [
    "DJI.xlsx", "HSI.xlsx", "KOSPI.xlsx", "LSE.xlsx",
    "NASDAQ.xlsx", "NIFTY50.xlsx", "NYSE.xlsx",
    "RUSSELL2000.xlsx", "SENSEX.xlsx", "SP500.xlsx", "SSE.xlsx"
]


def scale_with_params(data, x_min, x_max, scaling):
    scaled = (data - x_min) / (x_max - x_min + 1e-8)
    scaled = scaled * scaling
    return scaled


if __name__ == "__main__":

    configs = get_all_configs()
    results = []

    for config_id, config in enumerate(configs):

        print(f"\nConfig {config_id}: {config}")

        for file in datasets:

            path = os.path.join(DATASET_FOLDER, file)
            prices = load_dataset(path)

            prices = prices.reshape(-1, 1)

            X, y = create_dataset(prices, config["window"], config["k"])

            # split
            X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

            x_min = X_train.min()
            x_max = X_train.max()

            X_train = scale_with_params(X_train, x_min, x_max, config["input_scaling"])
            X_val   = scale_with_params(X_val, x_min, x_max, config["input_scaling"])
            X_test  = scale_with_params(X_test, x_min, x_max, config["input_scaling"])

            y_train = scale_with_params(y_train, x_min, x_max, config["input_scaling"])
            y_val   = scale_with_params(y_val, x_min, x_max, config["input_scaling"])
            y_test  = scale_with_params(y_test, x_min, x_max, config["input_scaling"])

            start = time.time()
            model, ridge_models = train_model(X_train, y_train, config)
            training_time = time.time() - start

            pred = predict(model, ridge_models, X_val, config)

            y_val_orig = inverse_scale(y_val, config["input_scaling"], x_min, x_max).flatten()
            pred_orig  = inverse_scale(pred, config["input_scaling"], x_min, x_max)

            rmse, mae, mape = evaluate(y_val_orig, pred_orig)

            print(f"{file} | RMSE: {rmse:.3f}")

            results.append([
                config_id, file,
                config["window"], config["k"],
                config["hidden_size"], config["num_layers"],
                config["ridge_alpha"], config["input_scaling"],
                rmse, mae, mape, training_time
            ])

    columns = [
        "Config ID", "Dataset",
        "Window", "K",
        "Hidden", "Layers",
        "Ridge", "Scaling",
        "RMSE", "MAE", "MAPE",
        "Time"
    ]

    pd.DataFrame(results, columns=columns).to_csv("k_isolate.csv", index=False)

    print("\nDone.")