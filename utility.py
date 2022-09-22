from torch.utils.data import Dataset, DataLoader
from torch.distributions import MultivariateNormal
import torch.nn as nn
import pandas as pd
import torch

class Datasets(Dataset):

    def __init__(self, dataset):
        if dataset == "boom":
            df = pd.read_csv("datasets/boomerang.csv")
        elif dataset == "blobs":
            df = pd.read_csv("datasets/two_blobs.csv")
        else:
            df = pd.read_csv("datasets/two_moons.csv")
        features = df.iloc[:,0:2].values
        self.features=torch.tensor(features, dtype=torch.float32)
 
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self,idx):
        return self.features[idx]
    
class TrainingLoss(nn.Module):
    """
    From https://arxiv.org/pdf/1912.02762v2.pdf section 2.3.1 equation 14
    """
    def __init__(self, dim):
        super().__init__()
        self.mvn = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    
    def forward(self, z0, inverse_log_det):
        log_prob_z0 = self.mvn.log_prob(z0)
        return -((log_prob_z0 + inverse_log_det).mean())