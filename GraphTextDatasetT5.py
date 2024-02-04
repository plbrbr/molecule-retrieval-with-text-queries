import os
import os.path as osp
import torch
from torch_geometric.data import Dataset 
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset
import pandas as pd

from graph_augmentation import *

class GraphTextDatasetT5(Dataset):
    def __init__(self, root, gt, split, tokenizer=None, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.description = pd.read_csv(os.path.join(self.root, split+'.tsv'), sep='\t', header=None)   
        self.description = self.description.set_index(0).to_dict()
        self.cids = list(self.description[1].keys())
        
        self.idx_to_cid = {idx: cid for idx, cid in enumerate(self.cids)}
        super(GraphTextDatasetT5, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_t5/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
      edge_index  = []
      x = []
      with open(raw_path, 'r') as f:
        next(f)
        for line in f: 
          if line != "\n":
            edge = *map(int, line.split()), 
            edge_index.append(edge)
          else:
            break
        next(f)
        for line in f: #get mol2vec features:
          substruct_id = line.strip().split()[-1]
          if substruct_id in self.gt.keys():
            x.append(self.gt[substruct_id])
          else:
            x.append(self.gt['UNK'])
        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        task_definition = 'Definition: You are given a molecule description in English. Your job is to generate the molecule SELFIES that fits the description.\n\n'
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            task_input = task_definition + f'Now complete the following example -\nInput: {self.description[1][cid]}\nOutput: '   
            text_input = self.tokenizer([task_input],
                                   return_tensors="pt", 
                                   truncation=True, 
                                   max_length=512,
                                   padding="max_length",
                                   add_special_tokens=True,)
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index, input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data 