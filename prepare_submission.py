from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
from sklearn.metrics.pairwise import cosine_similarity
from utils import prepare_submission, contrastive_loss
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda")

nb_epochs = 20
batch_size = 128
learning_rate = 2e-5

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = Model(
    model_name=model_name, 
    num_node_features=300, 
    nout=768, 
    nhid=1024, 
    graph_hidden_channels=1024
).to(device) # nout = bert model hidden dim

optimizer = optim.AdamW(
    model.parameters(), 
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

save_path = os.path.join('./', 'model.pt')

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

with torch.no_grad():
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
