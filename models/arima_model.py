from statsmodels.tsa.arima.model import ARIMA

def train(series):
    model = ARIMA(series, order=(1,1,1))
    return model.fit()

def predict(model, steps):
    return model.forecast(steps=steps)