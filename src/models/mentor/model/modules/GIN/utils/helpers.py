import torch

def expand_left(src, dim, dims):
    for _ in range(dims + dim if dim < 0 else dim):
        src = src.unsqueeze(0)
    return src
