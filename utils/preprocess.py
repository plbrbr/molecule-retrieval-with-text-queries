import subprocess
import sys
    
sys.path.append('Kaggle_dataset/source_code')

from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch

if __name__ == '__main__':
    model_name = 'allenai/scibert_scivocab_cased'
    nb_epochs = 10
    batch_size = 32

    if not (len(sys.argv) in [1, 4]):
        print("Uncorrect number of arguments, please provide model_name, nb_epochs, batch_size and learning_rate")
        exit()

    else:
        if len(sys.argv) == 4:
            model_name = sys.argv[1]
            nb_epochs = sys.argv[2]
            batch_size = sys.argv[3]

            with open('Kaggle_kernel/kaggle-script.py', 'r') as f:
                old_file = f.readlines()

            try:
                old_file[55] = "    model_name = '{}'\n".format(model_name)
                old_file[63] = "    nb_epochs = {}\n".format(nb_epochs)
                old_file[64] = "    batch_size = {}\n".format(batch_size)
                print("Script successfully changed")
            except:
                print("Script could not be changed")

            with open('Kaggle_kernel/kaggle-script.py', 'w') as f:
                f.writelines(old_file)

        print(f"Running new experiment with model_name='{model_name}', nb_epochs={nb_epochs}, batch_size={batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("Kaggle_dataset/data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='Kaggle_dataset/data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='Kaggle_dataset/data/', gt=gt, split='train', tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_loader = DataLoader(val_dataset, batch_size=int(batch_size), shuffle=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, num_workers=4)

    test_cids_dataset = GraphDataset(root='Kaggle_dataset/data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='Kaggle_dataset/data/test_text.txt', tokenizer=tokenizer)

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    test_loader = DataLoader(test_cids_dataset, batch_size=int(batch_size), shuffle=False)