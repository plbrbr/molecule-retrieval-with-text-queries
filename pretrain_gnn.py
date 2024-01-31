from dataloader import PreTrainGraphDataset
from torch_geometric.data import DataLoader
from Model import GraphEncoder
from utils import loss_cl, normalise_embeddings_using_train
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import time
from copy import deepcopy

writer = SummaryWriter()

gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
gt = normalise_embeddings_using_train(gt)
train_dataset = PreTrainGraphDataset(root='./data/', gt=gt, split='train')

train_dataset1 = train_dataset.shuffle()
train_dataset2 = deepcopy(train_dataset1)

device = torch.device("cuda")

nb_epochs = 200
batch_size = 128
learning_rate = 2e-4

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

loss = 0
losses = []
time1 = time.time()
printEvery = 50
saveEvery = 50

for epoch in range(1, nb_epochs+1):
    print(f"-----EPOCH{epoch}-----")
    model.train()
    for iter, (graph_batch1, graph_batch2) in enumerate(zip(train_loader1, train_loader2)):
        x_graph1 = model(graph_batch1.to(device))
        x_graph2 = model(graph_batch2.to(device))

        current_loss = loss_cl(x_graph1, x_graph2)
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        current_loss = current_loss.detach().cpu().item()
        loss += current_loss

        writer.add_scalar("train/loss", current_loss, iter * epoch)

        if iter % printEvery == 0:
            loss = loss / printEvery
            time2 = time.time()
            print(f"Iteration: {iter}, Time: {(time2 - time1):.4f} s, training loss: {(loss):.4f}")
            losses.append(loss)
            loss = 0

        if iter > 0 and iter % saveEvery == 0:  
            save_path = f"./gnn_pretrained_{iter}.pt"
            torch.save({
                'epoch': nb_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,
            }, save_path)
            print(f"checkpoint saved to: {save_path}")

writer.flush()
writer.close()
