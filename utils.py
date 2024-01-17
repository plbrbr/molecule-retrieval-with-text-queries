import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

TML = torch.nn.TripletMarginWithDistanceLoss(
   distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)
)
def triplet_loss(x1, x2):
    return TML(x1, x2, torch.roll(x2, 1, 0)) + TML(x2, x1, torch.roll(x1, 1, 0))

def prepare_submission(similarity, path: str):
    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
    solution.to_csv(path, index=False)
