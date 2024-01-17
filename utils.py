import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(x1, x2):
    logits = F.cosine_similarity(x1[:, None, :], x2[None, :, :], dim=2)
    labels = torch.arange(logits.shape[0], device=x1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def prepare_submission(similarity, path: str):
    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
    solution.to_csv(path, index=False)