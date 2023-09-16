import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch_scatter import scatter
from torchmetrics import Accuracy, AUROC, F1Score, ConfusionMatrix
from torchcontrib.optim import SWA
import pytorch_lightning as pl

from .modules.ATTENTION.conv import GlobalAttention
from .modules.PGNN.conv import PGNN
from .modules.GIN.conv import GINConv
from .modules.GAT.conv import GATv2Conv
from .modules.inits import *


# Topology channel.
class TopologyChannel(nn.Module):
    """Architecture of the 'topology' channel."""

    def __init__(self, input_dim, hidden_dim, num_layers, dropout, flow_conv, agg_conv, agg_team, gat, init_func):
        super().__init__()

        self.agg_team = agg_team
        self.gat = gat

        if self.gat:
            self.conv_GAT = GATv2Conv(input_dim, hidden_dim, flow=flow_conv, aggr=agg_conv, add_self_loops=True,
                                      init_func=init_func)
            input_dim = hidden_dim

        self.convs = ModuleList()
        for i in range(num_layers):
            mlp = Sequential(Linear(input_dim, hidden_dim),
                             BatchNorm1d(hidden_dim),
                             ReLU(inplace=True),
                             Dropout(p=dropout),
                             Linear(hidden_dim, hidden_dim),
                             BatchNorm1d(hidden_dim),
                             ReLU(inplace=True))
            conv = GINConv(mlp, aggr=agg_conv, train_eps=True, flow=flow_conv, normalize=False,
                           init_func=init_func)
            self.convs.append(conv)
            input_dim = hidden_dim

    def forward(self, x, edge_index, batch):
        att = None
        if self.gat:
            x, att = self.conv_GAT(x, edge_index, return_attention_weights=True)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=att[1], self_loops_weights=True) if self.gat else conv(x, edge_index)
        x = scatter(x, batch, dim=0, reduce=self.agg_team)
        return x, att


# Centrality channel.
class CentralityChannel(nn.Module):
    """Architecture of the 'centrality' channel."""

    def __init__(self, input_dim, hidden_dim, num_layers, flow_conv, agg_conv, dropout, init_func):
        super().__init__()

        self.convs = ModuleList()
        for _ in range(num_layers):
            mlp = Sequential(Linear(input_dim, hidden_dim),
                             BatchNorm1d(hidden_dim),
                             ReLU(inplace=True),
                             Dropout(p=dropout),
                             Linear(hidden_dim, hidden_dim),
                             BatchNorm1d(hidden_dim),
                             ReLU(inplace=True))
            conv = GINConv(mlp, aggr=agg_conv, train_eps=True, flow=flow_conv, normalize=False,
                           init_func=init_func)
            self.convs.append(conv)
            input_dim = hidden_dim

    def forward(self, x, edge_index, edge_weight, mask_teams):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
        return x[mask_teams]


# Position channel.
class PositionChannel(nn.Module):
    """Architecture of the 'position' channel."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_anchorsets, num_layers, agg_conv, use_dropout,
                 dropout, init_func):
        super().__init__()

        self.conv = PGNN(input_dim, hidden_dim, output_dim, n_anchorsets, num_layers, aggr=agg_conv,
                         use_dropout=use_dropout, dropout=dropout, init_func=init_func)

    def forward(self, x, dists_max, dists_argmax, mask_teams):
        x = self.conv(x, dists_max, dists_argmax)
        return x[mask_teams]


# Single framework.
class SingleFramework(pl.LightningModule):
    def __init__(self, input_dim_t: int, input_dim_c: int, input_dim_p: int, out_dim: int, n_anchorsets: int,
                 hidden_dim: int = 64, flow_conv_t: str = "source_to_target", flow_conv_c: str = "source_to_target",
                 agg_conv_t: str = "sum", agg_conv_c: str = "sum", agg_conv_p: str = "sum", agg_team_t: str = "sum",
                 use_dropout_p: bool = False, dropout_t: float = 0.5, dropout_c: float = 0.5, dropout_p: float = 0.5,
                 dropout_a: float = 0.5, epochs: int = 100, lr_base: float = 0.01, lr_swa: float = 0.005,
                 swa_start: float = 0.75, swa_freq: int = 5, gat_t: bool = True, topology: bool = True,
                 centrality: bool = True, position: bool = True, tensorboard: SummaryWriter = None,
                 init_func: torch.Tensor = None, **kwargs):
        """
        Architecture of the 'Single Framework' composed by three channels: topology, centrality and position.

        :param input_dim_t: the input layer dimension of the topology channel
        :param input_dim_c: the input layer dimension of the centrality channel
        :param input_dim_p: the input layer dimension of the position channel
        :param out_dim: the output layer dimension of the framework (number of classes to predict)
        :param n_anchorsets: the number of anchor sets for the P-GNN algorithm of the position channel
        :param hidden_dim: the hidden layer dimension for all the activated channels
        :param flow_conv_t: the direction of the convolutional flow for the topology channel; allowed values are
            'source_to_target' and 'target_to_source'
        :param flow_conv_c: the direction of the convolutional flow for the centrality channel; allowed values are
            'source_to_target' and 'target_to_source'
        :param agg_conv_t: the type of messages aggregation for topology channel; allowed values are 'sum', 'mean',
            'max', 'min'
        :param agg_conv_c: the type of messages aggregation for centrality channel; allowed values are 'sum', 'mean',
            'max', 'min'
        :param agg_conv_p: the type of messages aggregation for position channel; allowed values are 'sum', 'mean',
            'max', 'min'
        :param agg_team_t: the type of aggregation for topology channel from node level to team level; allowed values
            are 'sum', 'mean', 'max', 'min'
        :param use_dropout_p: whether to activate the dropout for the position channel
        :param dropout_t: the value of the dropout for the topology channel
        :param dropout_c: the value of the dropout for the centrality channel
        :param dropout_p: the value of the dropout for the position channel
        :param dropout_a: the value of the dropout for the attention layer
        :param epochs: the numbers of training epochs
        :param lr_base: the value of the learning rate
        :param lr_swa: the value of the SWA learning rate
        :param swa_start: the start SWA
        :param swa_freq: the epochs frequency in computing SWA
        :param gat_t: whether to activate the attention GAT layer for the topology channel
        :param topology: whether to activate the topology channel or not
        :param centrality: whether to activate the centrality channel or not
        :param position: whether to activate the position channel or not
        :param tensorboard: a tensorboard instance for storing training information
        :param init_func: the type of weights' initialization
        :param kwargs:
        :return:
        """
        super().__init__()

        # In order to use swa correctly we need to turn off automatic optimization.
        self.automatic_optimization = False

        # Learning SWA parameters.
        self.lr_base = lr_base
        self.lr_swa = lr_swa
        self.swa_start = int(swa_start * epochs)
        self.swa_freq = swa_freq
        # Last epoch for swa batch norm update.
        self.last_epoch = epochs - 1

        # Tensorboard.
        self.tensorboard = tensorboard

        # Channels.
        self.channels = []
        if topology:
            self.topology = TopologyChannel(input_dim=input_dim_t, hidden_dim=hidden_dim, num_layers=3,
                                            dropout=dropout_t, flow_conv=flow_conv_t, agg_conv=agg_conv_t,
                                            agg_team=agg_team_t, gat=gat_t, init_func=init_func)
            self.channels.append("topology")
        if centrality:
            self.centrality = CentralityChannel(input_dim=input_dim_c, hidden_dim=hidden_dim, num_layers=3,
                                                flow_conv=flow_conv_c, agg_conv=agg_conv_c, dropout=dropout_c,
                                                init_func=init_func)
            self.channels.append("centrality")
        if position:
            self.position = PositionChannel(input_dim=input_dim_p, hidden_dim=hidden_dim, output_dim=hidden_dim,
                                            n_anchorsets=n_anchorsets, num_layers=2, agg_conv=agg_conv_p,
                                            use_dropout=use_dropout_p, dropout=dropout_p,
                                            init_func=init_func)
            self.channels.append("position")

        self.n_channels = len(self.channels)

        assert self.n_channels > 0

        # Attention.
        if self.n_channels > 1:
            self.attention = Sequential(Linear(hidden_dim, hidden_dim),
                                        BatchNorm1d(hidden_dim),
                                        ReLU(inplace=True),
                                        Dropout(p=dropout_a),
                                        Linear(hidden_dim, 1))

            self.aggregation = GlobalAttention(self.attention, init_func=init_func)

        # Classifier.
        self.classifier = Sequential(Linear(hidden_dim, out_dim))
        reset(self.classifier, init_func=init_func)

        # Attention coefficients for the channels.
        self.attentions_channels = None
        # Attention coefficients for the edges into topology channel.
        self.attentions_topology = None

        # Evaluation metrics.
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.test_f1 = F1Score(num_classes=out_dim)
        self.test_auroc = AUROC(num_classes=out_dim)
        self.test_confmtx = ConfusionMatrix(num_classes=out_dim)

    def forward(self, batch):
        # Channels.
        x = []
        if "topology" in self.channels:
            x_1, self.attentions_topology = self.topology(batch["topology"].x_norm, batch["topology"].edge_index,
                                                          batch["topology"].composition)
            x_1 = F.normalize(x_1, p=2, dim=-1)
            x.append(x_1)
        if "centrality" in self.channels:
            x_2 = self.centrality(batch["centrality"].x, batch["centrality"].edge_index,
                                  batch["centrality"].edge_weight,
                                  batch["centrality"].mask_teams)
            x_2 = F.normalize(x_2, p=2, dim=-1)
            x.append(x_2)
        if "position" in self.channels:
            x_3 = self.position(batch["position"].x, batch["position"].dists_max, batch["position"].dists_argmax,
                                batch["position"].mask_teams)
            x_3 = F.normalize(x_3, p=2, dim=-1)
            x.append(x_3)

        if self.n_channels > 1:
            # Batch attention channels.
            batch_att = torch.arange(0, end=x[0].shape[0], device="cuda").repeat(self.n_channels)

            # Skip layer.
            x_skip = torch.cat([x_i.unsqueeze(0) for x_i in x])
            x_skip = torch.sum(x_skip, 0)

            # Concatenation embedding channels.
            x = torch.cat(x, 0)
            # Attention sum.
            x, att_channels = self.aggregation(x, batch_att)
            # Reshape: [n_teams, n_channels].
            self.attentions_channels = att_channels.T.reshape(
                (att_channels.shape[0] // self.n_channels, self.n_channels)[::-1]).T

            x = x + x_skip
        else:
            x = x[0]

        return self.classifier(x), x

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        # Forward.
        y_hat, _ = self(batch)

        # Evaluation metrics.
        train_loss = F.cross_entropy(y_hat[batch.mask_train], batch.y[batch.mask_train])
        train_acc = self.train_acc(y_hat.softmax(dim=-1)[batch.mask_train], batch.y[batch.mask_train])

        self.log("train_acc", train_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.manual_backward(train_loss)
        opt.step()

        # Tensorboard.
        if self.tensorboard is not None:
            self.tensorboard.add_scalar("train_accuracy", train_acc, self.current_epoch)
            self.tensorboard.add_scalar("train_loss", train_loss, self.current_epoch)
            self.tensorboard.add_scalar("learning_rate", self.optimizers().param_groups[0]["lr"], self.current_epoch)

        # At the end of the last epoch swap the weights with swa and update the batch normalization statistics.
        if self.current_epoch == self.last_epoch:
            opt.swap_swa_sgd()
            self.update_bn(batch)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # Forward.
        y_hat, _ = self(batch)

        # Evaluation metrics.
        val_loss = F.cross_entropy(y_hat[batch.mask_val], batch.y[batch.mask_val])

        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        # Forward.
        y_hat, z = self(batch)

        # Evaluation metrics.
        test_loss = F.cross_entropy(y_hat[batch.mask_test], batch.y[batch.mask_test])
        test_acc = self.test_acc(y_hat.softmax(dim=-1)[batch.mask_test], batch.y[batch.mask_test])
        test_f1 = self.test_f1(y_hat.softmax(dim=-1).argmax(-1)[batch.mask_test], batch.y[batch.mask_test])
        test_auroc = self.test_auroc(y_hat.softmax(dim=-1)[batch.mask_test], batch.y[batch.mask_test])
        self.confusion_matrix = self.test_confmtx(y_hat.softmax(dim=-1).argmax(-1)[batch.mask_test],
                                                  batch.y[batch.mask_test]).cpu().numpy()

        self.test_predictions = (y_hat.softmax(dim=-1)[batch.mask_test].cpu().numpy(),
                                 batch.y[batch.mask_test].cpu().numpy())

        self.log("test_acc", test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_f1", test_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_auroc", test_auroc, prog_bar=True, on_step=False, on_epoch=True)

        # Tensorboard.
        if self.tensorboard is not None:
            self.tensorboard.add_embedding(z[batch.mask_test], metadata=batch.y[batch.mask_test].detach().cpu().numpy(),
                                           global_step=self.current_epoch, tag=f"epoch_{self.current_epoch}")

        return test_acc

    def configure_optimizers(self):
        base_opt = torch.optim.Adam(self.parameters(), lr=self.lr_base)
        opt = SWA(base_opt, swa_start=self.swa_start, swa_freq=self.swa_freq, swa_lr=self.lr_swa)
        return opt

    @torch.no_grad()
    def update_bn(self, batch):
        momenta = {}
        for module in self.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            return

        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0

        self(batch)

        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
