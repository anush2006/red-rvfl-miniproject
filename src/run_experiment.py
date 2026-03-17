import numpy as np
import yfinance as yf
import torch
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

from red_revfl_orchestrator import RedRVFLOrchestrator
from metrics import rmse, mae, mape


# -------------------------------
# DATASET LOCATION
# -------------------------------

dataset_folder = "../RVFL_Datasets"

datasets = [
    "DJI.xlsx",
    "HSI.xlsx",
    "KOSPI.xlsx",
    "LSE.xlsx",
    "NASDAQ.xlsx",
    "NIFTY50.xlsx",
    "NYSE.xlsx",
    "RUSSELL2000.xlsx",
    "SENSEX.xlsx",
    "SP500.xlsx",
    "SSE.xlsx"
]


# -------------------------------
# CREATE TIME WINDOW DATA
# -------------------------------

def create_dataset(data, window=10):
    X = []
    y = []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)


# -------------------------------
# RUN EXPERIMENT
# -------------------------------

results = []

for file in datasets:

    print("\nRunning dataset:", file)

    path = os.path.join(dataset_folder, file)

    df = pd.read_excel(path, skiprows=2)
    df.columns = ['Date', 'Close']
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    prices = df['Close'].values

    if "Close" in df.columns:
        prices = df["Close"].values
    else:
        prices = df.iloc[:,1].values

    # -------------------------------
    # SAVE MIN/MAX BEFORE SCALING
    # -------------------------------

    x_min = prices.min()
    x_max = prices.max()

    # -------------------------------
    # NORMALIZATION
    # -------------------------------

    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

    # -------------------------------
    # CREATE WINDOW DATA
    # -------------------------------

    X, y = create_dataset(prices_scaled, window=10)

    # -------------------------------
    # TRAIN TEST SPLIT
    # -------------------------------

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test  = X[split:]
    y_train = y[:split]
    y_test  = y[split:]

    # -------------------------------
    # CONVERT TO TENSOR
    # -------------------------------

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)

    # -------------------------------
    # INITIALIZE REDRVFL MODEL
    # -------------------------------

    model = RedRVFLOrchestrator(
        input_features=1,
        hidden_size=50,
        num_layers=3
    )

    # -------------------------------
    # EXTRACT FEATURE MATRICES
    # -------------------------------

    feature_matrices = model.extract_features(X_train_tensor)

    # -------------------------------
    # TRAIN RIDGE REGRESSION
    # -------------------------------

    ridge_models = []
    for D in feature_matrices:
        ridge = Ridge(alpha=0.1)
        ridge.fit(D, y_train)
        ridge_models.append(ridge)

    # -------------------------------
    # PREDICT
    # -------------------------------

    pred = model.predict(X_test_tensor, ridge_models)

    # -------------------------------
    # INVERSE TRANSFORM BEFORE METRICS
    # -------------------------------

    y_test_orig = (y_test * (x_max - x_min) + x_min).flatten()
    pred_orig   = pred   * (x_max - x_min) + x_min

    # -------------------------------
    # METRICS (on original scale)
    # -------------------------------

    rmse_val = rmse(y_test_orig, pred_orig)
    mae_val  = mae(y_test_orig,  pred_orig)
    mape_val = mape(y_test_orig, pred_orig)

    print("RMSE:", rmse_val)
    print("MAE :", mae_val)
    print("MAPE:", mape_val)

    results.append([file, rmse_val, mae_val, mape_val])


# -------------------------------
# FINAL RESULTS TABLE
# -------------------------------

results_df = pd.DataFrame(results, columns=["Dataset", "RMSE", "MAE", "MAPE"])

print("\nFINAL RESULTS\n")
print(results_df)

results_df.to_csv("results.csv", index=False)