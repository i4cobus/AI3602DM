import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(in_dim, emb_dim)

    def forward(self, x) -> torch.Tensor:
        x = self.emb(x)
        return x


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        pass
