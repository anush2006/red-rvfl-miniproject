import numpy as np
from sklearn.linear_model import Ridge

class edRVFL:
    def __init__(self, input_dim, hidden_dim=100, layers=3):
        self.W = [np.random.randn(input_dim, hidden_dim)]
        for _ in range(1, layers):
            self.W.append(np.random.randn(input_dim + hidden_dim, hidden_dim))
        self.models = []

    def fit(self, X, y):
        H_prev = None
        for W in self.W:
            if H_prev is None:
                H = np.tanh(X @ W)
            else:
                H = np.tanh(np.concatenate([H_prev, X], axis=1) @ W)

            D = np.concatenate([H, X], axis=1)
            model = Ridge(alpha=0.1).fit(D, y)
            self.models.append(model)
            H_prev = H

    def predict(self, X):
        preds = []
        H_prev = None
        for W, model in zip(self.W, self.models):
            if H_prev is None:
                H = np.tanh(X @ W)
            else:
                H = np.tanh(np.concatenate([H_prev, X], axis=1) @ W)

            D = np.concatenate([H, X], axis=1)
            preds.append(model.predict(D))
            H_prev = H

        return np.median(preds, axis=0)