import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "models"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import numpy as np
import pandas as pd
import torch
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

# ========================
# IMPORT YOUR MODELS
# ========================
from persistence_model import predict as persistence_predict
from models.svr_model import train as svr_train, predict as svr_predict
from models.arima_model import train as arima_train, predict as arima_predict
from models.lstm_model import LSTMModel, train as lstm_train
from models.gru_model import GRUModel, train as gru_train
from models.tcn_model import TCN
from models.rvfl_model import RVFL
from models.edrvfl_model import edRVFL
from models.edesn_model import edESN
from models.vmd_lstm_model import build_model as vmd_lstm_build
from models.ewtrvfl_model import build_model as ewtrvfl_build
from models.ewtedrvfl_model import build_model as ewtedrvfl_build

# Proposed model
from src.red_revfl_orchestrator import RedRVFLOrchestrator
from src.metrics import rmse, mae, mape

# ========================
# SETTINGS
# ========================
np.random.seed(42)
torch.manual_seed(42)

dataset_folder = "RVFL_Datasets"

datasets = [
    "DJI.xlsx","HSI.xlsx","KOSPI.xlsx","LSE.xlsx",
    "NASDAQ.xlsx","NIFTY50.xlsx","NYSE.xlsx",
    "RUSSELL2000.xlsx","SENSEX.xlsx","SP500.xlsx","SSE.xlsx"
]

WINDOW = 10

# ========================
# DATASET CREATION
# ========================
def create_dataset(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

# ========================
# MAIN LOOP
# ========================
results = []

for file in datasets:

    print("\n====================")
    print("Dataset:", file)
    print("====================")

    path = os.path.join(dataset_folder, file)
    df = pd.read_excel(path, skiprows=2)

    df.columns = ['Date', 'Close']
    df = df.dropna()
    prices = df['Close'].values

    x_min, x_max = prices.min(), prices.max()

    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1,1))

    X, y = create_dataset(prices_scaled, WINDOW)

    n = len(X)

    train_end = int(n * 0.7)
    val_end   = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    y_train = y_train.ravel()
    y_test  = y_test.ravel()

    # flatten for ML models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    # tensors for DL
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    def evaluate(name, pred):
        y_true = (y_test * (x_max - x_min) + x_min).flatten()
        y_pred = pred * (x_max - x_min) + x_min

        results.append([
            file, name,
            rmse(y_true, y_pred),
            mae(y_true, y_pred),
            mape(y_true, y_pred)
        ])

    # ========================
    # MODELS
    # ========================

    # 1. Persistence
    pred = persistence_predict(X_test)
    evaluate("Persistence", pred)

    # 2. SVR
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.svm import SVR

    param_dist = {
        "C": np.logspace(-2, 3, 10),
        "epsilon": [0.001, 0.01, 0.1],
        "gamma": ["scale", "auto"]
    }

    svr = SVR()
    search = RandomizedSearchCV(svr, param_dist, n_iter=10, cv=3, n_jobs=-1)
    search.fit(X_train_flat, y_train)

    model = search.best_estimator_
    pred = model.predict(X_test_flat)
    evaluate("SVR", pred)

    # 3. ARIMA
    train_series = prices_scaled[:train_end + WINDOW].flatten()

    model = arima_train(train_series)
    pred = arima_predict(model, len(y_test))
    pred = pred[:len(y_test)]
    pred = arima_predict(model, len(y_test))
    pred = pred[:len(y_test)]
    evaluate("ARIMA", pred)

    # 4. LSTM
    model = LSTMModel()
    model = lstm_train(model, X_train_tensor, y_train_tensor)
    pred = model(X_test_tensor).detach().numpy().flatten()
    evaluate("LSTM", pred)

    # 5. GRU
    model = GRUModel()
    model = gru_train(model, X_train_tensor, y_train_tensor)
    pred = model(X_test_tensor).detach().numpy().flatten()
    evaluate("GRU", pred)

    # 6. TCN
    model = TCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for _ in range(50):
        pred = model(X_train_tensor)
        loss = loss_fn(pred.squeeze(), y_train_tensor.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pred = model(X_test_tensor).detach().numpy().flatten()
    evaluate("TCN", pred)

    # 7. RVFL
    model = RVFL(X_train_flat.shape[1])
    model.fit(X_train_flat, y_train)
    pred = model.predict(X_test_flat)
    evaluate("RVFL", pred)

    # 8. edRVFL
    model = edRVFL(X_train_flat.shape[1])
    model.fit(X_train_flat, y_train)
    pred = model.predict(X_test_flat)
    evaluate("edRVFL", pred)

    # 9. edESN
    model = edESN(X_train_flat.shape[1])
    model.fit(X_train_flat, y_train)
    pred = model.predict(X_test_flat)
    evaluate("edESN", pred)

    # 10. VMD-LSTM (placeholder)
    model = vmd_lstm_build()
    model = lstm_train(model, X_train_tensor, y_train_tensor)
    pred = model(X_test_tensor).detach().numpy().flatten()
    evaluate("VMD-LSTM", pred)

    # 11. EWTRVFL
    model = ewtrvfl_build(X_train_flat.shape[1])
    model.fit(X_train_flat, y_train)
    pred = model.predict(X_test_flat)
    evaluate("EWTRVFL", pred)

    # 12. EWTedRVFL
    model = ewtedrvfl_build(X_train_flat.shape[1])
    model.fit(X_train_flat, y_train)
    pred = model.predict(X_test_flat)
    evaluate("EWTedRVFL", pred)

    # 13. Proposed RedRVFL
    red = RedRVFLOrchestrator(
        input_features=1,
        hidden_size=50,
        num_layers=3
    )

    feature_matrices = red.extract_features(X_train_tensor)

    ridge_models = []
    for D in feature_matrices:
        ridge = Ridge(alpha=0.1)
        ridge.fit(D, y_train)
        ridge_models.append(ridge)

    pred = red.predict(X_test_tensor, ridge_models)
    evaluate("RedRVFL", pred)


# ========================
# SAVE RESULTS
# ========================
results_df = pd.DataFrame(
    results,
    columns=["Dataset","Model","RMSE","MAE","MAPE"]
)

# ========================
# PAPER STYLE TABLES
# ========================

print("\n===== TABLE 3: RMSE RESULTS =====\n")

rmse_table = results_df.pivot(index="Dataset", columns="Model", values="RMSE")

# reorder columns like paper
model_order = [
    "ARIMA", "Persistence", "SVR", "TCN", "LSTM", "GRU",
    "RVFL", "EWTRVFL", "VMD-LSTM", "edESN", "edRVFL", "EWTedRVFL", "RedRVFL"
]

rmse_table = rmse_table[model_order]

print(rmse_table.round(3))


print("\n===== TABLE 4: MAE RESULTS =====\n")

mae_table = results_df.pivot(index="Dataset", columns="Model", values="MAE")
mae_table = mae_table[model_order]

print(mae_table.round(3))


print("\n===== TABLE 5: MAPE RESULTS =====\n")

mape_table = results_df.pivot(index="Dataset", columns="Model", values="MAPE")
mape_table = mape_table[model_order]

print(mape_table.round(4))

results_df.to_csv("all_model_results.csv", index=False)