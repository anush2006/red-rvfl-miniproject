from sklearn.svm import SVR

def train(X, y):
    model = SVR()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)