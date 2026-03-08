import torch
import torch.nn as nn
import numpy as np

"""
Architecture Module
This module defines the RandomLSTM feature extractor and the construction
of the RVFL feature matrix used by the RedRVFL architecture.

Data Flow
1. Input Window
   The input tensor `xtensor` has shape:

       (batch_size, window_size, input_features)

   In this project:
       input_features = 1  (closing price only)

   Example:
       (128, 96, 1)

2. Random LSTM Feature Extraction
   The input window is passed through a randomly initialized LSTM
   whose weights are frozen (no training occurs).

   The LSTM processes the sequence and outputs the final hidden state.

       hidden_tensor shape:
       (batch_size, hidden_size)

   Example:
       (128, 64)

3. Flatten Input Window
   The original time window is flattened:

       (batch_size, window_size, input_features)
       → (batch_size, window_size * input_features)

   Example:
       (128, 96)

4. Feature Matrix Construction

   The hidden representation and flattened input are concatenated:

       D = [hidden_features , flattened_input]

   Final feature matrix shape:

       (batch_size, hidden_size + window_size * input_features)

   Example:
       hidden_size = 64
       window_size = 96
       input_features = 1

       D shape:
       (128, 160)

Purpose
-------

The RandomLSTM acts as a fixed nonlinear feature extractor, while
the RVFL feature matrix `D` is used as the input to a Ridge Regression
model that learns the final prediction weights.
"""


# Random LSTM feature extractor
class RandomLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Freeze all LSTM parameters
        for param in self.lstm.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x):

        with torch.no_grad():
            # output, (h_n, c_n)
            _, (h_n, _) = self.lstm(x)

        # h_n shape = (num_layers, batch_size, hidden_size)
        # selecting last layer hidden state
        hidden_state = h_n[-1]  # shape: (batch_size, hidden_size)

        return hidden_state


def flatten_window(X):
    """
    Flatten input time window.

    Input:
        torch tensor
        (batch_size, window_size, features)

    Output:
        numpy array
        (batch_size, window_size * features)
    """

    batch_size = X.shape[0]

    xflat = X.reshape(batch_size, -1)

    # convert to numpy for ridge regression
    return xflat.detach().cpu().numpy()


def build_layer_input(previous_hidden, xtensor):
    """
    Construct input for deeper RedRVFL layers.

    Layer l receives:
        [h_(l-1) , X]

    previous_hidden:
        torch tensor
        (batch_size, hidden_size)

    xtensor:
        torch tensor
        (batch_size, window_size, features)

    returns:
        torch tensor
        (batch_size, window_size, features + hidden_size)
    """

    window_size = xtensor.shape[1]

    # expand hidden representation across time dimension
    hidden_expanded = previous_hidden.unsqueeze(1).repeat(1, window_size, 1)

    # concatenate along feature dimension
    layer_input = torch.cat([xtensor, hidden_expanded], dim=2)

    return layer_input


def build_feature_matrix(xtensor, hidden_tensor):
    """
    Build RVFL feature matrix.

    D = [hidden_features , flattened_input]

    Input:
        xtensor:
            (batch_size, window_size, features)

        hidden_tensor:
            (batch_size, hidden_size)

    Output:
        D:
            numpy array
            (batch_size, hidden_size + window_size * features)
    """

    # flatten original input
    xflat = flatten_window(xtensor)

    # convert hidden features to numpy
    hidden_np = hidden_tensor.detach().cpu().numpy()

    # concatenate features
    D = np.concatenate([hidden_np, xflat], axis=1)

    return D