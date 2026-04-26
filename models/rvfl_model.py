import numpy as np
from sklearn.linear_model import Ridge

class RVFL:
    def __init__(self, input_dim, hidden_dim=100):
        self.W = np.random.randn(input_dim, hidden_dim)
        self.beta = None

    def _hidden(self, X):
        return np.tanh(X @ self.W)

    def fit(self, X, y):
        H = self._hidden(X)
        D = np.concatenate([H, X], axis=1)
        self.beta = Ridge(alpha=0.1).fit(D, y)

    def predict(self, X):
        H = self._hidden(X)
        D = np.concatenate([H, X], axis=1)
        return self.beta.predict(D)