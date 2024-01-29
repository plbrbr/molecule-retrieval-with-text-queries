import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(x1, x2):
    logits = torch.matmul(x1, torch.transpose(x2, 0, 1))
    #logits = torch.sum((x1[:, None, :] - x2[None, :, :])**2, dim=2)
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
        margin=0.3,
    distance_function=distance_function
)
def triplet_loss(x1, x2):
    distances = distance_function(x1[:, None, :], x2[None, :, :], dim=2)
    x1_negatives = x2[torch.argmax(distances, dim=1)]
    x2_negatives = x1[torch.argmax(distances, dim=0)]
    return TML(x1, x2, x1_negatives) + TML(x2, x1, x2_negatives)

def prepare_submission(similarity, path: str):
    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col != 'ID']]
    solution.to_csv(path, index=False)

def loss_cl(x1, x2, tau=0.1):
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss