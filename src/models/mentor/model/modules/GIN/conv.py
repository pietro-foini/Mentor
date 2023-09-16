import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import matmul

from .message_passing import MessagePassing
from ..inits import reset


class GINConv(MessagePassing): 
    """How Powerful are Graph Neural Networks?

    Args:
        nn (torch.nn.Module): A neural network that
            maps node features `x` of shape `[-1, in_channels]` to
            shape `[-1, out_channels]`, *e.g.*, defined by
            `torch.nn.Sequential`.
        eps (float, optional): (Initial) `\epsilon`-value.
            (default: :obj:`0.`).
        train_eps (bool, optional): If set to `True`, `\epsilon`
            will be a trainable parameter. (default: `False`).
        **kwargs (optional): Additional arguments of
            `torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn, eps = 0., train_eps = False, normalize = False, init_func = None, **kwargs): 
        kwargs.setdefault("aggr", "sum") 
        super(GINConv, self).__init__(**kwargs) 
        self.nn = nn 
        self.initial_eps = eps         
        self.normalize = normalize

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps])) # So in general: buffers = ‘fixed tensors / non-learnable parameters / stuff that does not require gradient’; parameters = ‘learnable parameters, requires gradient’. Quindi in questo caso questo parametro non contribuisce alla backpropagation.
        self.reset_parameters(init_func) 

    def reset_parameters(self, init_func):
        reset(self.nn, init_func) 
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, size = None, **kwargs):              
        if isinstance(x, Tensor):
            # We put the features / embedding in a tuple where the first refers to x_j and the second to x_i.
            x = (x, x)
        
        # Self-loop weights (if weights are provided!).
        add_self_weights = kwargs.get("self_loops_weights", False) 
        if add_self_weights:
            edge_weight = kwargs.get("edge_weight")
            # Last number of edges weights are referred to self-loop connections.
            edge_weight, self_loop_weight = torch.split(edge_weight, [edge_index.shape[1], 
                                                                      edge_weight.shape[0] - edge_index.shape[1]], dim = 0)
            kwargs.update({"edge_weight": edge_weight})
        
        # In this part convolution with neighbors is done.
        out = self.propagate(edge_index, size, x=x, **kwargs) 

        x_r = x[1] # x_i.
        if x_r is not None:
            if add_self_weights:
                out += ((1 + self.eps) * x_r) * self_loop_weight
            else:
                out += (1 + self.eps) * x_r
        
        if self.normalize:
            x = F.normalize(self.nn(out), p=2, dim=-1)
        else:
            x = self.nn(out)

        return x

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x): 
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr) 
    