import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, TopKPooling, GINConv
from GINConvWithGraphNorm import GINConvWithGraphNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class GraphEncoderWithAttention(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, n_gins=3, dropout_rate=0.2):
        super(GraphEncoderWithAttention, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.dropout_rate = dropout_rate
        self.activation = nn.SiLU()

        self.gcn1 = GATv2Conv(num_node_features, graph_hidden_channels // 2)
        self.gcn2 = GATv2Conv(graph_hidden_channels // 2, graph_hidden_channels // 2)
        #self.gcn3 = GATv2Conv(graph_hidden_channels // 2, graph_hidden_channels // 2)
        #self.pool1 = TopKPooling(graph_hidden_channels // 2, ratio=0.8)
        #self.pool2 = TopKPooling(graph_hidden_channels // 2, ratio=0.8)
        #self.pool3 = TopKPooling(graph_hidden_channels // 2, ratio=0.8)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        #self.dropout3 = nn.Dropout(dropout_rate)

        self.last_linear = nn.Sequential(
            nn.Linear(graph_hidden_channels, nhid),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(nhid, nhid),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(nhid, nhid),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(nhid, nout),
        )

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.activation(self.gcn1(x, edge_index)) 
        x = self.dropout1(x)
        #x, edge_index, edge_attr, batch, perm, score = self.pool1(x, batch)
        #x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = self.activation(self.gcn2(x, edge_index))
        x = self.dropout2(x)
        #x, edge_index, edge_attr, batch, perm, score = self.pool2(x, batch)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        
        #x = self.activation(self.gcn3(x, edge_index))
        #x = self.dropout3(x)
        #x, edge_index, edge_attr, batch, perm, score = self.pool3(x, batch)
        #x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        #x = x1 + x2 #+ x3

        #x, edge_index, edge_attr, batch, perm, score = self.pool1(x, batch)
        x = self.last_linear(x)
        return x
