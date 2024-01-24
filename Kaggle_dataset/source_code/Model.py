from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, n_gins=5):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.gins = nn.ModuleList()
        self.gins.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(num_node_features, graph_hidden_channels),
                    nn.BatchNorm1d(graph_hidden_channels), nn.ReLU(),
                    nn.Linear(graph_hidden_channels, graph_hidden_channels), nn.ReLU()
                ), train_eps=True)
        )
        for _ in range(n_gins-1):
            self.gins.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(graph_hidden_channels, 2*graph_hidden_channels),
                        nn.BatchNorm1d(2*graph_hidden_channels), nn.ReLU(),
                        nn.Linear(2*graph_hidden_channels, graph_hidden_channels), nn.ReLU()
                    ), train_eps=True)) 
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        for gin in self.gins:
            x = gin(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, model_name):
        self.d_hidden = 768
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Sequential(
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.LeakyReLU(), nn.Dropout(p=0.2),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.LeakyReLU()
        )
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = encoded_text.last_hidden_state[:, 0, :]
        last_hidden_state = self.linear(last_hidden_state)
        #print(encoded_text.last_hidden_state.size())
        return last_hidden_state
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
