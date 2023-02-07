import torch
import random
import torch.nn as nn
from sparse_graph import SparseGraph


def get_negative_tests(G: SparseGraph, size: int) -> torch.Tensor:
    n_nodes = G.coo.shape[0]
    negatives = []
    for i in range(size):
        src = choose_a_node(n_nodes)
        neighbours = G.get_neighbours(src)
        dst = choose_a_node(n_nodes)
        while dst == src or dst in neighbours:
            dst = choose_a_node(n_nodes)
        negatives.append([src, dst])

    return torch.tensor(negatives)


def choose_a_node(high, low=0) -> int:
    return random.randint(low, high-1)


def normalized_cosine_similiarty(x: torch.Tensor, y: torch.Tensor):
    # cosine_sim = torch.cosine_similarity(x, y)
    # cosine_sim = cosine_sim - cosine_sim.min()
    # cosine_sim = cosine_sim / cosine_sim.max()

    # return cosine_sim
    return (torch.cosine_similarity(x, y) + 1) / 2


def calc_auc(D0: torch.LongTensor, D1: torch.LongTensor, model: nn.Module):
    model.eval()
    pred0 = model.forward(D0)  # B,2,Emb
    pred1 = model.forward(D1)  # B,2,Emb
    prob0 = normalized_cosine_similiarty(pred0[:, 0, :], pred0[:, 1, :])
    prob1 = normalized_cosine_similiarty(pred1[:, 0, :], pred1[:, 1, :])
    prob1_ext = prob1.repeat_interleave(prob0.shape[0])
    prob0_ext = prob0.repeat(prob1.shape[0])
    auc = torch.sum(prob0_ext < prob1_ext).float()\
        / prob0.shape[0] / prob1.shape[0]
    return auc
