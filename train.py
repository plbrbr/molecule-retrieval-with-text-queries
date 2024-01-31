from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score 
from utils import prepare_submission, contrastive_loss, normalise_embeddings_using_train
import numpy as np
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import time
import os

writer = SummaryWriter()

#model_name = 'bert-base-uncased'
#model_name = 'allenai/scibert_scivocab_uncased'
model_name = "thenlper/gte-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
gt = normalise_embeddings_using_train(gt)
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda")

nb_epochs = 50
batch_size = 128

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = Model(
    model_name=model_name, 
    num_node_features=300, 
    nout=512, # nout = bert model hidden dim
    nhid=512, 
    graph_hidden_channels=512
)

gnn_checkpoint = torch.load("./gnn_pretrained.pt")
model.graph_encoder.load_state_dict(gnn_checkpoint['model_state_dict'])
del gnn_checkpoint

# checkpoint = torch.load("./model.pt")
# model.load_state_dict(checkpoint['model_state_dict'])
# del checkpoint

model.to(device)

optimizer = optim.AdamW(
    [
        {'params': model.parameters(), 'lr': 4e-4},
    ],
    betas=(0.9, 0.999),
    weight_decay=0.01
)
#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.8, verbose=True)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=2e-4, 
    steps_per_epoch=len(train_loader), 
    epochs=nb_epochs
)

loss = 0
losses = []
time1 = time.time()
printEvery = 50
best_validation_accuracy = 0.0

for epoch in range(1, nb_epochs+1):
    print('-----EPOCH{}-----'.format(epoch))
    model.train()
    for iter, batch in enumerate(train_loader):
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        
        x_graph, x_text = model(
            graph_batch.to(device), 
            input_ids.to(device), 
            attention_mask.to(device)
        )
        contrast_loss = contrastive_loss(x_graph, x_text)
        # trip_loss = triplet_loss(x_graph, x_text)
        current_loss = contrast_loss # + trip_loss
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()

        lr_scheduler.step()
        writer.add_scalar("train/loss", loss, iter * epoch)
        writer.add_scalar("train/lr", lr_scheduler.get_last_lr(), iter * epoch)

        if iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0

    model.eval()
    graph_embeddings, text_embeddings = [], []
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            x_graph, x_text = model(
                graph_batch.to(device), 
                input_ids.to(device), 
                attention_mask.to(device)
            )
            # compute the val accuracy
            graph_embeddings.append(x_graph.detach())
            text_embeddings.append(x_text.detach())

            current_loss = contrastive_loss(x_graph, x_text)   
            val_loss += current_loss.item()
    val_loss = val_loss/len(val_loader)

    graph_embeddings = torch.vstack(graph_embeddings)
    text_embeddings = torch.vstack(text_embeddings)
    similarity = cosine_similarity(text_embeddings.cpu(), graph_embeddings.cpu())
    y_true = np.eye(similarity.shape[0])
    val_accuracy = label_ranking_average_precision_score(y_true, similarity)

    writer.add_scalar("test/loss", val_loss, len(train_loader) * epoch)
    writer.add_scalar("test/score", val_accuracy, len(train_loader) * epoch)

    best_validation_accuracy = max(best_validation_accuracy, val_accuracy)
    print(f"-----EPOCH{epoch}----- done.  Validation loss: {val_loss:.3f}, Score: {val_accuracy:.3f}")
    if best_validation_accuracy == val_accuracy:
        print('validation score improved saving checkpoint...')
        save_path = os.path.join('./', 'model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': val_loss,
            'validation_accuracy': val_accuracy,
            'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))
    
writer.flush()
writer.close()
