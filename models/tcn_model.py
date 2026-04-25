import torch
import torch.nn as nn

class TCN(nn.Module):
    def __init__(self, input_size=1, channels=32):
        super().__init__()
        self.conv = nn.Conv1d(input_size, channels, kernel_size=3, padding=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x[:, :, -1]
        return self.fc(x)