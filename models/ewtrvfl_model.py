# Simplified placeholder (EWT not implemented)
from models.rvfl_model import RVFL

def build_model(input_dim):
    return RVFL(input_dim)