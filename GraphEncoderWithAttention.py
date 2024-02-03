import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, TopKPooling
from GINConvWithGraphNorm import GINConvWithGraphNorm
from torch_geometric.nn import global_mean_pool, global_add_pool

class GraphEncoderWithAttention(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, n_gins=3, dropout_rate=0.1):
        super(GraphEncoderWithAttention, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.dropout_rate = dropout_rate

        self.gcn1 = GATv2Conv(num_node_features, graph_hidden_channels)
        self.gcn2 = GATv2Conv(graph_hidden_channels, graph_hidden_channels)
        self.gcn3 = GATv2Conv(graph_hidden_channels, graph_hidden_channels)
        self.pool1 = TopKPooling(graph_hidden_channels, ratio=0.8)

        self.last_linear = nn.Sequential(
            nn.Linear(graph_hidden_channels, nhid),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(nhid, nhid),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(nhid, nout),
        )

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = F.relu(self.gcn1(x, edge_index)) 
        x = F.relu(self.gcn2(x, edge_index)) 
        x = F.relu(self.gcn3(x, edge_index))
        x = self.pool1(x, batch)
        x = global_add_pool(x, batch)
        x = self.last_linear(x)
        return x
