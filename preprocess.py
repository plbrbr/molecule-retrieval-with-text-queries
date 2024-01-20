import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
    
sys.path.append('Kaggle_dataset/source_code')

from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch

if __name__ == '__main__':

    model_name = 'bert-base-cased'
    if len(sys.argv):
        model_name = sys.argv[1]

    ## edit Kaggle_kernel/kaggle-script.py
    with open('Kaggle_kernel/kaggle-script.py', 'r') as f:
        old_file = f.readlines()

    old_file[old_file.index("    model_name = 'bert-base-cased'\n")] = "    model_name = '{}'\n".format(model_name)

    with open('Kaggle_kernel/kaggle-script.py', 'w') as f:
        f.writelines(old_file)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("Kaggle_dataset/data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='Kaggle_dataset/data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='Kaggle_dataset/data/', gt=gt, split='train', tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_cids_dataset = GraphDataset(root='Kaggle_dataset/data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='Kaggle_dataset/data/test_text.txt', tokenizer=tokenizer)

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)