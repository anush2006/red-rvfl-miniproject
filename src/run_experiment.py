import numpy as np
import torch
import pandas as pd
import time
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

from red_revfl_orchestrator import RedRVFLOrchestrator
from metrics import rmse, mae, mape


# -------------------------------
# REPRODUCIBILITY
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------
# DATASET PATH
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_folder = os.path.abspath(os.path.join(BASE_DIR, "..", "RVFL_Datasets"))

datasets = [
    "DJI.xlsx", "HSI.xlsx", "KOSPI.xlsx", "LSE.xlsx",
    "NASDAQ.xlsx", "NIFTY50.xlsx", "NYSE.xlsx",
    "RUSSELL2000.xlsx", "SENSEX.xlsx", "SP500.xlsx", "SSE.xlsx"
]

WINDOW_SIZE_OPTIONS = [48, 96, 124]
HIDDEN_SIZE_OPTIONS = [20, 50, 100, 128]
NUM_LAYER_OPTIONS = [1, 3, 5, 7]
RIDGE_ALPHA_OPTIONS = [0.0, 0.01, 0.1, 0.5, 1.0]
INPUT_SCALING_OPTIONS = [0.1, 0.3, 0.5, 0.7, 1.0]


def get_all_configs():
    configs = []

    for window in WINDOW_SIZE_OPTIONS:
        for hidden in HIDDEN_SIZE_OPTIONS:
            for layers in NUM_LAYER_OPTIONS:
                for alpha in RIDGE_ALPHA_OPTIONS:
                    for scaling in INPUT_SCALING_OPTIONS:

                        configs.append({
                            "window": window,
                            "hidden_size": hidden,
                            "num_layers": layers,
                            "ridge_alpha": alpha,
                            "input_scaling": scaling
                        })

    return configs


# -------------------------------
# DATA PREP
# -------------------------------
def create_dataset(data, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)


def scale_data(data, scaling_factor):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))
    return scaled * scaling_factor, scaler


# -------------------------------
# MAIN EXPERIMENT LOOP
# -------------------------------
if __name__ == "__main__":

    configs = get_all_configs()
    print(f"Total configs: {len(configs)}")

    results = []

    for config_id, config in enumerate(configs):

        print("\n==============================")
        print(f"Config ID: {config_id}")
        print(config)

        for file in datasets:

            path = os.path.join(dataset_folder, file)

            if not os.path.exists(path):
                print(f"Missing: {file}")
                continue

            df = pd.read_excel(path, skiprows=2, engine="openpyxl")
            df.columns = ['Date', 'Close']
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close'])

            prices = df['Close'].values

            x_min = prices.min()
            x_max = prices.max()

            # Scaling
            scaling = max(config["input_scaling"], 1e-8)
            prices_scaled, _ = scale_data(prices, scaling)

            # Windowing
            X, y = create_dataset(prices_scaled, window=config["window"])

            n = len(X)

            train_end = int(n * 0.7)
            val_end   = int(n * 0.8)

            # Train
            X_train = X[:train_end]
            y_train = y[:train_end]

            # Validation
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]

            # Test
            X_test = X[val_end:]
            y_test = y[val_end:]

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            # Model
            model = RedRVFLOrchestrator(
                input_features=1,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"]
            )

            # Training
            start_time = time.time()

            feature_matrices = model.extract_features(X_train_tensor)

            ridge_models = []
            for D in feature_matrices:
                ridge = Ridge(alpha=config["ridge_alpha"])
                ridge.fit(D, y_train)
                ridge_models.append(ridge)

            training_time = time.time() - start_time

            # Prediction
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            pred = model.predict(X_val_tensor, ridge_models)
            pred = pred.flatten()
            
            # Inverse scaling
            y_val_orig = (y_val / scaling) * (x_max - x_min) + x_min
            y_val_orig = y_val_orig.flatten()
            pred_orig   = (pred   / scaling) * (x_max - x_min) + x_min

            # Metrics
            rmse_val = rmse(y_val_orig, pred_orig)
            mae_val  = mae(y_val_orig, pred_orig)
            mape_val = mape(y_val_orig, pred_orig)

            print(f"{file} | RMSE: {rmse_val:.3f}")

            results.append([
                config_id,
                file,
                config["window"],
                config["hidden_size"],
                config["num_layers"],
                config["ridge_alpha"],
                config["input_scaling"],
                rmse_val,
                mae_val,
                mape_val,
                training_time
            ])

        # -------------------------------
        # SAVE AFTER EACH CONFIG
        # -------------------------------
        results_df = pd.DataFrame(results, columns=[
            "Config ID", "Dataset",
            "Window", "Hidden Size", "Num Layers",
            "Ridge Alpha", "Input Scaling",
            "RMSE", "MAE", "MAPE", "Training Time"
        ])

        results_df.to_csv("results_partial.csv", index=False)

    # Final save
    results_df.to_csv("results.csv", index=False)

    print("All experiments completed.")