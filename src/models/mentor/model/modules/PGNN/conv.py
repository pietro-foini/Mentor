import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inits import reset


class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, init_func):
        super(PGNN_layer, self).__init__()
        
        self.input_dim = input_dim

        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim, 1)        
        self.act = nn.ReLU()
        
        self.reset_parameters(init_func) 
        
    def reset_parameters(self, init_func):
        reset(self.modules(), init_func) 

    def forward(self, feature, dists_max, dists_argmax, agg_conv):
        # We take the features (attributes at first and embedding for the next layers) of the nodes of the anchorsets (so we flatten the matrix 'dists_argmax' thus obtaining a tensor of shape (n_nodes, n_anchorsets) where there could be repeated values) which have maximum distance from the nodes of the graph. In the obtained reshaped features matrix of shape (n_nodes, n_anchorsets, n_features) there can consequently be equal rows because a node of an anchorset could have maximum distance for more nodes of the graph from that anchorset.
        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1], feature.shape[1]))

        # Message computation function F that combines feature information of two nodes with their network distance: matrix M of anchorset messages, where each row i is an anchorset message Mi computed by F.
        messages = subset_features * dists_max.unsqueeze(-1)
        # messages: shape (n_nodes, n_anchorsets, n_features).

        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1) # Tensor of dimension (n_nodes, n_anchorsets, n_features). 
        messages = torch.cat((messages, self_feature), dim=-1) 
        # messages: shape (n_nodes, n_anchorsets, 2*n_features).
        
        # Hidden layer w of the paper. 
        messages = self.linear_hidden(messages).squeeze() 
        messages = self.act(messages) # shape (n_nodes, n_anchorsets, dimension hidden layer).

        out_position = self.linear_out_position(messages).squeeze(-1)  # shape (n_nodes, n_anchorsets). It's zv of the algorithm of the paper. The message from each node includes distances that reveal node positions as well as feature-based information from input node features.
        
        # Define some aggregation functions.
        AGGs = {"sum": torch.sum, "mean": torch.mean, "max": torch.max, "min": torch.min}
        
        if agg_conv in ["min", "max"]:
            out_structure = AGGs[agg_conv](messages, dim=1)[0] # n*d (n_nodes, hidden_dim). It's hv of the algorithm of the paper. P-GNNs also compute structure-aware messages, hv, which are computed via an order-invariant message aggregation function that aggregates messages across anchorsets, and are then fed into the next P-GNN layer as input. Aggregating them then makes them invariant with respect to the number of anchor sets actually used.
        else:
            out_structure = AGGs[agg_conv](messages, dim=1)  

        return out_position, out_structure

    
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, init_func):
        super(Nonlinear, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        
        self.reset_parameters(init_func) 
        
    def reset_parameters(self, init_func):
        reset(self.modules(), init_func) 

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class PGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_anchorsets, num_layers=2, aggr="mean", use_dropout=False, 
                 dropout=0.5, init_func=None, **kwargs):
        super(PGNN, self).__init__()
        
        self.aggr = aggr
        assert self.aggr in ["sum", "mean", "max", "min"]
        
        self.num_layers = num_layers
        self.use_dropout = use_dropout
        self.dropout = dropout
        
        if self.num_layers == 1:
            hidden_dim = output_dim
            
        # Layer PGNN.
        self.conv_first = PGNN_layer(input_dim, hidden_dim, init_func)
        
        # Add layers PGNN.
        if self.num_layers > 1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim, init_func) for i in range(num_layers - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim, init_func)
            
        # Last layer to reshape output.
        self.linear_out = nn.Linear(n_anchorsets, output_dim)
        
        self.reset_parameters(init_func) 
        
    def reset_parameters(self, init_func):
        reset(self.modules(), init_func) 

    def forward(self, x, dists_max, dists_argmax):
        # First layer P-GNN.
        x_position, x = self.conv_first(x, dists_max, dists_argmax, self.aggr) # x: the features matrix; dists_max: a tensor of dimension (n_nodes, n_anchorsets) where the maximum distance between the node in question (row) and the anchorset in question (column) are entered; dists_argmax: a tensor of dimension (n_nodes, n_anchorsets) where the id of the node of the corresponding anchorset (based on the column) to which corresponds the maximum distance from the reference node (the one on the rows) is inserted.

        if self.num_layers == 1:
            x_position = self.linear_out(x_position)
            return x_position
        
        x = F.relu(x) # Note: optional!
        if self.use_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers-2):
            # Note that unlike GNNs, we cannot feed the output embeddings zv from the previous layer to the next layer. The deeper reason we cannot feed zv to the next layer is that the position of a node is always relative to the chosen anchor-sets.
            _, x = self.conv_hidden[i](x, dists_max, dists_argmax, self.aggr)
            x = F.relu(x) # Note: optional!
            if self.use_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_position, x = self.conv_out(x, dists_max, dists_argmax, self.aggr)
        x_position = F.normalize(x_position, p=2, dim=-1)
        
        x_position = self.linear_out(x_position)
        
        return x_position
