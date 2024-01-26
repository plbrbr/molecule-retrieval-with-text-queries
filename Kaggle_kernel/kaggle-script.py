import subprocess
import sys, os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
    
sys.path.append('/kaggle/input/KAGGLE-PBALMA/source_code')

from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import torch.nn.functional as F
from Model import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score 
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch import optim
import time
if __name__ == '__main__':
    torch.cuda.empty_cache()
    CE = torch.nn.CrossEntropyLoss()
    def contrastive_loss(x1, x2):
        logits = torch.matmul(x1, torch.transpose(x2, 0, 1))
        labels = torch.arange(logits.shape[0], device=x1.device)
        return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

    CE_none = torch.nn.CrossEntropyLoss(reduction='none')
    def contrastive_loss_component(x1, x2):
        logits = x1[:, None, :] * x2[None, :, :] 
        labels = torch.arange(logits.shape[0], device=x1.device)
        logits_sum = torch.sum(logits, dim=2)
        loss = CE(logits_sum, labels) + CE(torch.transpose(logits_sum, 0, 1), labels)

        logits = logits.T
        labels = labels.repeat((logits.shape[0], 1))
        loss += torch.sum(
            CE_none(logits, labels) + CE_none(torch.transpose(logits, 1, 2), labels)
        )
        return loss

    def distance_function(x, y, dim=1):
        return 1.0 - F.cosine_similarity(x, y, dim=dim)

    TML = torch.nn.TripletMarginWithDistanceLoss(
            margin=0.1,
        distance_function=distance_function
    )
    def triplet_loss(x1, x2):
        distances = distance_function(x1[:, None, :], x2[None, :, :], dim=2)
        x1_negatives = x2[torch.argmax(distances, dim=1)]
        x2_negatives = x1[torch.argmax(distances, dim=0)]
        return TML(x1, x2, x1_negatives) + TML(x2, x1, x2_negatives)

    model_name = 'allenai/scibert_scivocab_cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("../input/KAGGLE-PBALMA/data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='../input/KAGGLE-PBALMA/data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='../input/KAGGLE-PBALMA/data/', gt=gt, split='train', tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_epochs = 10
    batch_size = 32
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = Model(
        model_name=model_name, 
        num_node_features=300, 
        nout=768, 
        nhid=512, 
        graph_hidden_channels=256 
    ).to(device)

    optimizer = optim.AdamW(
        [{'params': model.graph_encoder.parameters(), 'lr': 1e-4},
         {'params': model.text_encoder.linear.parameters(), 'lr': 1e-4},
         {'params': model.text_encoder.bert.parameters(), 'lr': 2e-5}
        ],
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.2, verbose=True)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_accuracy = 0

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
        current_loss = contrastive_loss(x_graph, x_text) + triplet_loss(x_graph, x_text)

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()

        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 
    
    del graph_batch, x_graph, x_text, batch, input_ids, attention_mask
    torch.cuda.empty_cache()

    model.eval()     
    with torch.no_grad():
        graph_embeddings, text_embeddings = [], []  
        val_loss = 0        
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
            graph_embeddings.append(x_graph.detach())
            text_embeddings.append(x_text.detach())

            current_loss = contrastive_loss(x_graph, x_text)   
            val_loss += current_loss.item()

    del graph_batch, x_graph, x_text, batch, input_ids, attention_mask
    torch.cuda.empty_cache()

    graph_embeddings = torch.vstack(graph_embeddings)
    text_embeddings = torch.vstack(text_embeddings)
    similarity = cosine_similarity(text_embeddings.cpu(), graph_embeddings.cpu())
    y_true = np.eye(similarity.shape[0])
    val_accuracy = label_ranking_average_precision_score(y_true, similarity)
    scheduler.step(val_accuracy)
    best_validation_accuracy = max(best_validation_accuracy, val_accuracy)
    print(f"-----EPOCH{i+1}----- done.  Validation loss: {val_loss/len(val_loader):.3f}, score: {val_accuracy:.3f}")
    if best_validation_accuracy==val_accuracy:
        print('Validation score improved - saving checkpoint...')
        save_path = os.path.join('../working/', 'best_model.pt')
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
        }, save_path)
        print('Checkpoint saved to: {}'.format(save_path))
        
    del graph_embeddings, text_embeddings, similarity, y_true
    torch.cuda.empty_cache()
        
print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
with torch.no_grad():
    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()

    test_cids_dataset = GraphDataset(root='../input/KAGGLE-PBALMA/data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='../input/KAGGLE-PBALMA/data/test_text.txt', tokenizer=tokenizer)

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

    graph_embeddings = []
    for batch in test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())
    
    del batch, output

    test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
    text_embeddings = []
    for batch in test_text_loader:
        for output in text_model(batch['input_ids'].to(device), 
                                attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())
    
    del batch, output

similarity = cosine_similarity(text_embeddings, graph_embeddings)
solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv(os.path.join('../working/', 'submission.csv'), index=False)