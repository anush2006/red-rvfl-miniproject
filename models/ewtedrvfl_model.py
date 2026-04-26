# Simplified placeholder (EWT not implemented)
from models.edrvfl_model import edRVFL

def build_model(input_dim):
    return edRVFL(input_dim)