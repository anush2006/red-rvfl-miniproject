# Window size search space
WINDOW_SIZE_OPTIONS = [48, 96, 124]
# Batch size (fixed in paper)
BATCH_SIZE = 32
# LSTM latent units
HIDDEN_SIZE_RANGE = (20, 200)
# Number of layers
NUM_LAYER_RANGE = (1, 12)
# Ridge regularization
RIDGE_ALPHA_RANGE = (0.0, 1.0)
# Input scaling 
INPUT_SCALING_RANGE = (0.0, 1.0)