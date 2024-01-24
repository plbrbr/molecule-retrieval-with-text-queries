from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score 
from utils import prepare_submission, contrastive_loss, triplet_loss 
import numpy as np
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch import optim
import time
import os

#model_name = 'bert-base-uncased'
model_name = 'allenai/scibert_scivocab_uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda")

nb_epochs = 100
batch_size = 128
learning_rate = 2e-6

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = Model(
    model_name=model_name, 
    num_node_features=300, 
    nout=1024, # nout = bert model hidden dim
    nhid=1024, 
    graph_hidden_channels=1024
)

# checkpoint = torch.load("./model.pt")
# model.load_state_dict(checkpoint['model_state_dict'])
# del checkpoint

model.to(device)

optimizer = optim.AdamW(
    [
        {'params': model.graph_encoder.parameters(), 'lr': 1e-4},
        {'params': model.text_encoder.linear.parameters(), 'lr': 1e-4},
        {'params': model.text_encoder.bert.parameters(), 'lr': 2e-5}
    ],
    betas=(0.9, 0.999),
    weight_decay=0.01
)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer, 
    gamma=0.955,
    last_epoch=-1,
    verbose=True,
)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_accuracy = 0.0

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()
    for batch in train_loader:
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
        trip_loss = triplet_loss(x_graph, x_text)
        current_loss = contrast_loss + trip_loss
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()

        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f} {3:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery, trip_loss.item()))
            losses.append(loss)
            loss = 0
    lr_scheduler.step()
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
    graph_embeddings = torch.vstack(graph_embeddings)
    text_embeddings = torch.vstack(text_embeddings)
    similarity = cosine_similarity(text_embeddings.cpu(), graph_embeddings.cpu())
    y_true = np.eye(similarity.shape[0])
    val_accuracy = label_ranking_average_precision_score(y_true, similarity)
    best_validation_accuracy = max(best_validation_accuracy, val_accuracy)
    print(f"-----EPOCH{i+1}----- done.  Validation loss: {val_loss/len(val_loader):.3f}, Score: {val_accuracy:.3f}")
    if best_validation_accuracy == val_accuracy:
        print('validation score improved saving checkpoint...')
        save_path = os.path.join('./', 'model.pt')
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': val_loss,
            'validation_accuracy': val_accuracy,
            'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))


print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())

similarity = cosine_similarity(text_embeddings, graph_embeddings)
prepare_submission(similarity, "submissions/submission.csv")
