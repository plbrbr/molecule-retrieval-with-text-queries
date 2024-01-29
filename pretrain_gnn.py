from dataloader import PreTrainGraphDataset
from torch_geometric.data import DataLoader
from Model import GraphEncoder
from utils import loss_cl
import numpy as np
import torch
from torch import optim
import time
import os

from copy import deepcopy

gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = PreTrainGraphDataset(root='./data/', gt=gt, split='val') 
train_dataset = PreTrainGraphDataset(root='./data/', gt=gt, split='train')

train_dataset1 = train_dataset.shuffle()
train_dataset2 = deepcopy(train_dataset1)

device = torch.device("cuda")

nb_epochs = 100
batch_size = 64
learning_rate = 2e-4

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=False, num_workers=4)
train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=False, num_workers=4)

model = GraphEncoder(
    num_node_features=300, 
    nout=512, # nout = bert model hidden dim
    nhid=512, 
    graph_hidden_channels=512
)

# checkpoint = torch.load("./gnn_pretrained.pt")
# model.load_state_dict(checkpoint['model_state_dict'])
# del checkpoint

model.to(device)

optimizer = optim.AdamW(
    [
        {'params': model.parameters(), 'lr': learning_rate}
    ],
    betas=(0.9, 0.999),
    weight_decay=0.01
)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()
    for graph_batch1, graph_batch2 in zip(train_loader1, train_loader2):
        x_graph1 = model(graph_batch1.to(device))
        x_graph2 = model(graph_batch2.to(device))

        current_loss = loss_cl(x_graph1, x_graph2)
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()

        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0
    
    save_path = os.path.join('./', 'gnn_pretrained.pt')
    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print('checkpoint saved to: {}'.format(save_path))