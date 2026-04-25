import os
import numpy as np
import pandas as pd

from src.run.data_loader import load_dataset, create_dataset, split_data
from src.run.model_runner import train_model, predict
from src.run.evaluator import inverse_scale, evaluate


# -------------------------------
# PATHS
# -------------------------------
DATASET_FOLDER = "./RVFL_Datasets"
TOP_CONFIG_PATH = "top_10_configs.csv"


datasets = [
    "DJI.xlsx", "HSI.xlsx", "KOSPI.xlsx", "LSE.xlsx",
    "NASDAQ.xlsx", "NIFTY50.xlsx", "NYSE.xlsx",
    "RUSSELL2000.xlsx", "SENSEX.xlsx", "SP500.xlsx", "SSE.xlsx"
]


def scale_with_params(data, x_min, x_max, scaling):
    scaled = (data - x_min) / (x_max - x_min + 1e-8)
    scaled = scaled * scaling
    return scaled



configs_df = pd.read_csv(TOP_CONFIG_PATH)

results = []

for _, config in configs_df.iterrows():

    print(f"\nRunning FINAL config {config['Config ID']}")

    cfg = {
        "window": int(config["Window"]),
        "k": int(config["K"]),
        "hidden_size": int(config["Hidden Size"]),
        "num_layers": int(config["Num Layers"]),
        "ridge_alpha": float(config["Ridge Alpha"]),
        "input_scaling": float(config["Input Scaling"])
    }

    for file in datasets:

        path = os.path.join(DATASET_FOLDER, file)
        prices = load_dataset(path)

        # store original min/max for inverse scaling
        global_min, global_max = prices.min(), prices.max()

        # ❗ create dataset FIRST (no scaling yet)
        X, y = create_dataset(prices.reshape(-1, 1), cfg["window"], cfg["k"])

        # ❗ split BEFORE scaling
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

        # ❗ compute scaling from TRAIN ONLY
        x_min = X_train.min()
        x_max = X_train.max()

        # scale all splits consistently
        X_train = scale_with_params(X_train, x_min, x_max, cfg["input_scaling"])
        X_val   = scale_with_params(X_val, x_min, x_max, cfg["input_scaling"])
        X_test  = scale_with_params(X_test, x_min, x_max, cfg["input_scaling"])

        y_train = scale_with_params(y_train, x_min, x_max, cfg["input_scaling"])
        y_val   = scale_with_params(y_val, x_min, x_max, cfg["input_scaling"])
        y_test  = scale_with_params(y_test, x_min, x_max, cfg["input_scaling"])

        # combine train + val for final training (correct protocol)
        X_full = np.concatenate([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        # train
        model, ridge_models = train_model(X_full, y_full, cfg)

        # test
        pred = predict(model, ridge_models, X_test, cfg)

        # inverse scaling to original price space
        y_test_orig = inverse_scale(y_test, cfg["input_scaling"], x_min, x_max).flatten()
        pred_orig   = inverse_scale(pred, cfg["input_scaling"], x_min, x_max)

        # metrics
        rmse, mae, mape = evaluate(y_test_orig, pred_orig)

        print(f"{file} | RMSE: {rmse:.3f}")

        results.append([
            int(config["Config ID"]),
            file,
            cfg["window"], cfg["k"],
            cfg["hidden_size"], cfg["num_layers"],
            cfg["ridge_alpha"], cfg["input_scaling"],
            rmse, mae, mape
        ])


columns = [
    "Config ID", "Dataset",
    "Window", "K",
    "Hidden", "Layers",
    "Ridge", "Scaling",
    "RMSE", "MAE", "MAPE"
]

df = pd.DataFrame(results, columns=columns)
df.to_csv("final_results.csv", index=False)

print("\nFinal evaluation complete → saved as final_results.csv")