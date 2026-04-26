import numpy as np
from sklearn.linear_model import Ridge

class edESN:
    def __init__(self, input_dim, reservoir=100):
        self.W = np.random.randn(input_dim, reservoir)
        self.beta = None

    def _reservoir(self, X):
        return np.tanh(X @ self.W)

    def fit(self, X, y):
        H = self._reservoir(X)
        self.beta = Ridge(alpha=0.1).fit(H, y)

    def predict(self, X):
        H = self._reservoir(X)
        return self.beta.predict(H)