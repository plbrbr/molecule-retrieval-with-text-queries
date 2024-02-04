from dataloader import GraphTextDataset
from torch_geometric.data import DataLoader
from Model import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score 
from utils import prepare_submission, contrastive_loss, normalise_embeddings_using_train, triplet_loss
import numpy as np
from transformers import AutoTokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import time

nb_epochs = 20
batch_size = 75
learning_rate = 1e-4

writer = SummaryWriter()

#model_name = 'bert-base-uncased'
#model_name = 'allenai/scibert_scivocab_uncased'
model_name = "thenlper/gte-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
gt = normalise_embeddings_using_train(gt)
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda")

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

model = Model(
    model_name=model_name, 
    num_node_features=300, 
    nout=1024, # nout = bert model hidden dim
    nhid=1024, 
    graph_hidden_channels=512
)

gnn_checkpoint = torch.load("./gnn_pretrained.pt")
model.graph_encoder.load_state_dict(gnn_checkpoint['model_state_dict'])
del gnn_checkpoint

#checkpoint = torch.load("./model.pt")
#model.load_state_dict(checkpoint['model_state_dict'])
#del checkpoint

model.to(device)

optimizer = optim.AdamW(
    [
        {'params': model.parameters(), 'lr': learning_rate},
    ],
    betas=(0.9, 0.999),
    weight_decay=1e-2
)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.77)
#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.8, verbose=True)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=learning_rate, 
    steps_per_epoch=len(train_loader), 
    epochs=nb_epochs
)

loss = 0
losses = []
time1 = time.time()
printEvery = 50
best_validation_accuracy = 0.0

scaler = torch.cuda.amp.GradScaler()

for epoch in range(1, nb_epochs+1):
    print(f"-----EPOCH{epoch}-----")
    model.train()
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()

        n_iter = len(train_loader) * (epoch - 1) + iter + 1
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            x_graph, x_text = model(
                graph_batch.to(device), 
                input_ids.to(device), 
                attention_mask.to(device)
            )
            contrast_loss = contrastive_loss(x_graph, x_text)
            #trip_loss = triplet_loss(x_graph, x_text)
        current_loss = contrast_loss #+ trip_loss
        scaler.scale(current_loss).backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        current_loss = current_loss.item()
        loss += current_loss

        writer.add_scalars("loss", {"train": current_loss}, n_iter)
        writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], n_iter)

        if n_iter % printEvery == 0:
            loss = loss / printEvery
            time2 = time.time()
            print(f"Iteration: {n_iter}, Time: {(time2 - time1):.4f} s, training loss: {loss:.4f}")
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
    val_loss = val_loss / len(val_loader)

    graph_embeddings = torch.vstack(graph_embeddings)
    text_embeddings = torch.vstack(text_embeddings)
    similarity = cosine_similarity(text_embeddings.cpu(), graph_embeddings.cpu())
    y_true = np.eye(similarity.shape[0])
    val_accuracy = label_ranking_average_precision_score(y_true, similarity)
    #lr_scheduler.step()

    writer.add_scalars("loss", {"test": val_loss}, len(train_loader) * epoch)
    writer.add_scalar("test/score", val_accuracy, len(train_loader) * epoch)

    best_validation_accuracy = max(best_validation_accuracy, val_accuracy)
    print(f"-----EPOCH{epoch}----- done.  Validation loss: {val_loss:.3f}, Score: {val_accuracy:.3f}")
    if best_validation_accuracy == val_accuracy:
        print("validation score improved saving checkpoint...")
        save_path = "./model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': val_loss,
            'validation_accuracy': val_accuracy,
            'loss': loss,
        }, save_path)
        print(f"checkpoint saved to: {save_path}")
    
writer.flush()
writer.close()
