import torch
from torch import nn
import torch.nn.functional as F

from GINConvWithGraphNorm import GINConvWithGraphNorm
from torch_geometric.nn import global_mean_pool, global_add_pool

class GraphEncoderWithGraphNorm(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, n_gins=3, dropout_rate=0.1):
        super(GraphEncoderWithGraphNorm, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.dropout_rate = dropout_rate
        self.activation = nn.ReLU()
        self.gins = nn.ModuleList()
        self.gins.append(
            GINConvWithGraphNorm(
                [
                    nn.Linear(num_node_features, 2*graph_hidden_channels),
                    #nn.Linear(2*graph_hidden_channels, 2*graph_hidden_channels),
                    nn.Linear(2*graph_hidden_channels, 2*graph_hidden_channels),
                ], self.activation, train_eps=True)
        )
        for _ in range(n_gins-2):
            self.gins.append(
                GINConvWithGraphNorm(
                    [
                        nn.Linear(2*graph_hidden_channels, 2*graph_hidden_channels),
                        #nn.Linear(2*graph_hidden_channels, 2*graph_hidden_channels),
                        nn.Linear(2*graph_hidden_channels, 2*graph_hidden_channels),
                    ], self.activation, train_eps=True)
            )
        self.gins.append(
            GINConvWithGraphNorm(
                [
                    nn.Linear(2*graph_hidden_channels, 2*graph_hidden_channels),
                    #nn.Linear(2*graph_hidden_channels, 2*graph_hidden_channels),
                    nn.Linear(2*graph_hidden_channels, 2*graph_hidden_channels), 
                ], self.activation, train_eps=True)
        )
        self.last_linear = nn.Sequential(
            nn.Linear(2*graph_hidden_channels, nhid),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(nhid, nhid),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(nhid, nout),
        )

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        for gin in self.gins:
            x = gin(x, edge_index, batch)
        #x = global_mean_pool(x, batch)
        x = global_add_pool(x, batch)
        x = self.last_linear(x)
        return x
