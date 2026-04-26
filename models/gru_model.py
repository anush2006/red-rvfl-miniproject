import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

def train(model, X, y, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model