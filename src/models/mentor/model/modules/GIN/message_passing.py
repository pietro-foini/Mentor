from inspect import Parameter

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr

from .utils.helpers import expand_left
from .utils.inspector import Inspector


class MessagePassing(torch.nn.Module):
    """Base class for creating message passing layers.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (`"add"`, `"mean"`, `"max"` `None`).
            (default: `"add"`)
        flow (string, optional): The flow direction of message passing
            (`"source_to_target"` or `"target_to_source"`).
            (default: `"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: `-2`)
    """

    special_args = set(('edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size', 
                        'size_i', 'size_j', 'ptr', 'index', 'dim_size'))

    def __init__(self, aggr = "sum", flow = "source_to_target", node_dim = -2):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['sum', 'mean', 'max', 'min']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim

        self.inspector = Inspector(self) 
        # With the following commands you save the input parameters of the various functions that you pass to them in a dictionary with the key name of the function and the corresponding parameters as a value.
        self.inspector.inspect(self.message) # Ex: {"message": OrderedDict([('x_j', <Parameter "x_j">)])}.
        self.inspector.inspect(self.aggregate, pop_first=True) 
        self.inspector.inspect(self.message_and_aggregate, pop_first=True)
        self.inspector.inspect(self.update, pop_first=True)

        # It stores in a set some arguments of the various functions analyzed and of these that are passed to it (in truth they are repeated). However, arguments that are defined in 'self.special_args' are not considered.
        self.__user_args__ = self.inspector.keys(['message', 'aggregate', 'update']).difference(self.special_args)
        self.__fused_user_args__ = self.inspector.keys(['message_and_aggregate', 'update']).difference(self.special_args)

        # Support for "fused" message passing.
        self.fuse = self.inspector.implements('message_and_aggregate')

    def __check_input__(self, edge_index, size):
        the_size = [None, None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype == torch.long 
            assert edge_index.dim() == 2 # (2, n_edges).
            assert edge_index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size
        elif isinstance(edge_index, SparseTensor):
            if self.flow == 'target_to_source':
                raise ValueError(('Flow direction "target_to_source" is invalid for '
                                  'message propagation via `torch_sparse.SparseTensor`. If '
                                  'you really want to make use of a reverse message '
                                  'passing flow, pass in the transposed sparse tensor to '
                                  'the message passing module, e.g., `adj_t.t()`.'))
            the_size[0] = edge_index.sparse_size(1)
            the_size[1] = edge_index.sparse_size(0)
            return the_size

        raise ValueError(('`MessagePassing.propagate` only supports `torch.LongTensor` of '
                          'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
                          'argument `edge_index`.'))

    def __set_size__(self, size, dim, src):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return src.index_select(self.node_dim, index)
        elif isinstance(edge_index, SparseTensor):
            if dim == 1:
                rowptr = edge_index.storage.rowptr()
                rowptr = expand_left(rowptr, dim=self.node_dim, dims=src.dim())
                return gather_csr(src, rowptr)
            elif dim == 0:
                col = edge_index.storage.col()
                return src.index_select(self.node_dim, col)
        raise ValueError

    def __collect__(self, args, edge_index, size, kwargs):
        # First of all, the directionality of how the convolution is made is defined (obviously for an indirect network the choice is irrelevant).
        # That is, if the convolution is made with respect to the neighbors that point to you (source_to_target) or with respect to the neighbors that you point to (target_to_source).
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']: 
                out[arg] = kwargs.get(arg, Parameter.empty) 
            else:
                dim = 0 if arg[-2:] == '_j' else 1
                data = kwargs.get(arg[:-2], Parameter.empty) 

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2 
                    if isinstance(data[1-dim], Tensor):
                        self.__set_size__(size, 1-dim, data[1-dim]) 
                    data = data[dim] 

                if isinstance(data, Tensor): 
                    self.__set_size__(size, dim, data) 
                    # In this part it does a very important thing: it generates a tensor as long as there are edges in the network (n_edges, n_feature / embedding). It is the features / embedding of the nodes that point. For example, in the default configuration (source_to_target), if in the 'edge_index' we have that the first edge is the pair [12, 6], then the first value of this 'inputs' is the feature / embedding of 12 (of node 12 ).
                    data = self.__lift__(data, edge_index, j if arg[-2:] == '_j' else i)

                out[arg] = data

        if isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None
        elif isinstance(edge_index, SparseTensor):
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['ptr'] = edge_index.storage.rowptr()
            out['edge_weight'] = edge_index.storage.value() 
            out['edge_attr'] = edge_index.storage.value()
            out['edge_type'] = edge_index.storage.value()

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[1] or size[0]
        out['size_j'] = size[0] or size[1]
        out['dim_size'] = out['size_i']

        return out

    def propagate(self, edge_index, size, **kwargs):
        """The initial call to start propagating messages.

        Args:
            edge_index (Tensor or SparseTensor): A `torch.LongTensor` or a
                `torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape `[N, M]`.
                If `edge_index` is of type `torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in `edge_index[0]` are sent to
                nodes in `edge_index[1]`
                (in case `flow="source_to_target"`).
                If `edge_index` is of type
                `torch_sparse.SparseTensor`, its sparse indices
                `(row, col)` should relate to `row = edge_index[1]`
                and `col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                `propagate`.
            size (tuple, optional): The size `(N, M)` of the assignment
                matrix in case `edge_index` is a `LongTensor`.
                If set to `None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case `edge_index` is a
                `torch_sparse.SparseTensor`. (default: `None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index, size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute('message_and_aggregate', coll_dict)
            # Call the 'message_and_aggregate' function. If another class inherits from the current class then the 'message_and_aggregate' function of the reference class and not of this parent class will be used.
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            # Call the 'message' function. If another class inherits from the current class then the 'message' function (if it has it) of the reference class and not of this parent class will be used. In our case it calls the 'message' function of GINConv and not the one defined in this class.
            out = self.message(**msg_kwargs) 

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            # Call the 'aggregate' function. If another class inherits from the current class then the 'aggregate' function of the reference class will be used and not of this parent class. In our case, however, the GINConv class does not have the 'aggregate' function and therefore uses the 'aggregate' function defined in this class.
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

    def message(self, x_j):
        """Constructs messages from node `j` to node `i`
        in analogy to `\phi_{\mathbf{\Theta}}` for each edge in
        `edge_index`.
        This function can take any argument as input which was initially
        passed to `propagate`.
        Furthermore, tensors passed to `propagate` can be mapped to the
        respective nodes `i` and `j` by appending `_i` or
        `_j` to the variable name, *.e.g.* `x_i` and `x_j`.
        """
        return x_j

    def aggregate(self, inputs, index, edge_weight = None, ptr = None, dim_size = None):
        """Aggregates messages from neighbors as
        `\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to `propagate`.

        By default, this function will delegate its call to scatter functions
        that support "sum", "mean", "max" and "min" operations as specified in
        `__init__` by the `aggr` argument.
        """
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            # 'inputs' is a tensor as long as there are edges in the network (n_edges, n_feature / embedding). It is the features / embedding of the nodes that point. For example, in the default configuration (source_to_target), if in the 'edge_index' we have that the first edge is the pair [12, 6], then the first value of this 'inputs' is the feature / embedding of 12 (of node 12 ).
            # 'index' instead contains the indexes of the nodes to which the edges reach. For example in the above case we will have that the first value of 'index' is 6. This tensor can therefore contain repeated values if a node is pointed to by more nodes. Then through scatter the features / embedding of the nodes pointing to node 6 (in the default configuration) are aggregated according to the function defined in 'reduce'.
            # What is obtained in output is therefore a tensor along the number of nodes of the network in which the aggregate values (features / embedding) corresponding to the neighbors (who points, in the case of the default configuration) are stored. If a node is not pointed to by anyone it will not aggregate with anyone (in the default configuration).
            if edge_weight is not None:
                inputs = inputs * edge_weight
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)

    def message_and_aggregate(self, adj_t):
        """Fuses computations of `message` and `aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a `torch_sparse.SparseTensor`.
        """
        raise NotImplementedError

    def update(self, inputs):
        """Updates node embeddings in analogy to
        `\gamma_{\mathbf{\Theta}}` for each node
        `i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to `propagate`.
        """
        return inputs
