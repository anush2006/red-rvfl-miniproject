# model_runner.py

import torch
from sklearn.linear_model import Ridge
from src.red_revfl_orchestrator import RedRVFLOrchestrator


def train_model(X_train, y_train, config):

    window = config["window"]
    k = config["k"]

    X_train = X_train.reshape(X_train.shape[0], window * k, 1)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    model = RedRVFLOrchestrator(
        input_features=1,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"]
    )

    feature_matrices = model.extract_features(X_train_tensor)

    ridge_models = []
    for D in feature_matrices:
        ridge = Ridge(alpha=config["ridge_alpha"])
        ridge.fit(D, y_train)
        ridge_models.append(ridge)

    return model, ridge_models


def predict(model, ridge_models, X, config):

    window = config["window"]
    k = config["k"]

    X = X.reshape(X.shape[0], window * k, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    pred = model.predict(X_tensor, ridge_models)
    return pred.flatten()