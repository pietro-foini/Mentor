import math
import torch
import types


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn, init_func=None):
    def _reset(item, init_func):
        if hasattr(item, "reset_parameters"):
            if isinstance(item, torch.nn.Linear):
                if init_func is not None:
                    init_func(item.weight)
                    item.bias.data.fill_(0.0)
                else:
                    item.reset_parameters()

    if nn is not None:
        if hasattr(nn, "children") and len(list(nn.children())) > 0:  # If is a Sequential.
            for item in nn.children():
                _reset(item, init_func)
        elif isinstance(nn, types.GeneratorType):  # If self.modules()
            for item in nn:
                for item_child in item.children():
                    _reset(item_child, init_func)
        else:
            _reset(nn, init_func)  # If is a simple layer.
