import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, APPNP
from torch_sparse import SparseTensor, set_diag, spmm
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def create_gnn(hparams):
    if hparams['arch'] == "GAT":
        model = GAT(hparams)
    elif hparams['arch'] == "GATv2":
        model = GATv2(hparams)
    elif hparams['arch'] == "GCN":
        model = GCN(hparams)
    elif hparams['arch'] == "APPNP":
        model = APPNPNet(hparams)
    else:
        raise Exception("Not implemented")
    return model.to(hparams['device'])


class GAT(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        in_channels = hparams['in_channels']
        if hparams["smoothing_config"]["append_indicator"]:
            in_channels += 1
        self.conv1 = GATConv(in_channels,
                             hparams['hidden_channels'],
                             heads=hparams['k_heads'],
                             edge_dim=1,
                             dropout=hparams['conv_dropout'])
        self.conv2 = GATConv(hparams['k_heads']*hparams['hidden_channels'],
                             hparams['out_channels'],
                             edge_dim=1,
                             dropout=hparams['conv_dropout'])
        self.p_dropout = hparams['p_dropout']
        self.with_skip = hparams['with_skip']

        if not hparams['protected']:
            return

        if hparams['smoothing_config']['smoothing_distribution'] == "ablation":
            # ablation_token
            self.token = nn.Parameter(torch.zeros(hparams["in_channels"]))
            nn.init.xavier_uniform_(self.token.unsqueeze(0))

        self.empty = torch.empty(2, 0).long().to(hparams["device"])

    def forward(self, x_clean, x_noised, edge_idx):

        skip = 0
        if self.with_skip:
            hidden = F.relu(self.conv1(x_clean, self.empty))
            hidden = F.dropout(hidden, p=self.p_dropout,
                               training=self.training)
            skip = self.conv2(hidden, self.empty)

        hidden = F.elu(self.conv1(x_noised, edge_idx))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.conv2(hidden, edge_idx)

        return hidden + skip


class GATv2(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        in_channels = hparams['in_channels']
        if hparams["smoothing_config"]["append_indicator"]:
            in_channels += 1
        self.conv1 = GATv2Conv(in_channels,
                               hparams['hidden_channels'],
                               heads=hparams['k_heads'],
                               edge_dim=1,
                               dropout=hparams['conv_dropout'])
        self.conv2 = GATv2Conv(hparams['k_heads']*hparams['hidden_channels'],
                               hparams['out_channels'],
                               edge_dim=1,
                               dropout=hparams['conv_dropout'])
        self.p_dropout = hparams['p_dropout']
        self.with_skip = hparams['with_skip']

        if not hparams['protected']:
            return

        if hparams['smoothing_config']['smoothing_distribution'] == "ablation":
            # ablation_token
            self.token = nn.Parameter(torch.zeros(hparams["in_channels"]))
            nn.init.xavier_uniform_(self.token.unsqueeze(0))

        self.empty = torch.empty(2, 0).long().to(hparams["device"])

    def forward(self, x_clean, x_noised, edge_idx):

        skip = 0
        if self.with_skip:
            hidden = F.relu(self.conv1(x_clean, self.empty))
            hidden = F.dropout(hidden, p=self.p_dropout,
                               training=self.training)
            skip = self.conv2(hidden, self.empty)

        hidden = F.elu(self.conv1(x_noised, edge_idx))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.conv2(hidden, edge_idx)

        return hidden + skip


class GCN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        in_channels = hparams['in_channels']
        if hparams["smoothing_config"]["append_indicator"]:
            in_channels += 1
        self.conv1 = GCNConv(in_channels,
                             hparams['hidden_channels'],
                             dropout=hparams['conv_dropout'])
        self.conv2 = GCNConv(hparams['hidden_channels'],
                             hparams['out_channels'],
                             dropout=hparams['conv_dropout'])
        self.p_dropout = hparams['p_dropout']
        self.with_skip = hparams['with_skip']

        if not hparams['protected']:
            return

        if hparams['smoothing_config']['smoothing_distribution'] == "ablation":
            # ablation_token
            self.token = nn.Parameter(torch.zeros(hparams["in_channels"]))
            nn.init.xavier_uniform_(self.token.unsqueeze(0))

        self.empty = torch.empty(2, 0).long().to(hparams["device"])

    def forward(self, x_clean, x_noised, edge_idx):
        skip = 0
        if self.with_skip:
            hidden = F.relu(self.conv1(x_clean, self.empty))
            hidden = F.dropout(hidden, p=self.p_dropout,
                               training=self.training)
            skip = self.conv2(hidden, self.empty)

        hidden = F.relu(self.conv1(x_noised, edge_idx))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.conv2(hidden, edge_idx)
        return hidden + skip


class APPNPNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        in_channels = hparams['in_channels']
        if hparams["smoothing_config"]["append_indicator"]:
            in_channels += 1
        self.lin1 = nn.Linear(in_channels,
                              hparams["hidden_channels"],
                              bias=False)
        self.lin2 = nn.Linear(hparams["hidden_channels"],
                              hparams["out_channels"],
                              bias=False)
        # k_hops=10, appnp_alpha=0.15
        self.prop = APPNP(hparams["k_hops"], hparams["appnp_alpha"])
        self.p_dropout = hparams["p_dropout"]
        self.with_skip = hparams['with_skip']

        if not hparams['protected']:
            return

        if hparams['smoothing_config']['smoothing_distribution'] == "ablation":
            # ablation_token
            self.token = nn.Parameter(torch.zeros(hparams["in_channels"]))
            nn.init.xavier_uniform_(self.token.unsqueeze(0))

        self.empty = torch.empty(2, 0).long().to(hparams["device"])

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x_clean, x_noised):
        skip = 0
        if self.with_skip:
            hidden = F.relu(self.lin1(x_clean))
            hidden = F.dropout(hidden, p=self.p_dropout,
                               training=self.training)
            skip = self.lin2(hidden)

        hidden = F.relu(self.lin1(x_noised))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.lin2(hidden)
        hidden = self.prop(hidden)

        return hidden + skip
