# %%
import torch
import random
import sys
from loss import NegativeSamplingLoss
from metric import calc_auc, get_negative_tests
from model import SkipGram
from sparse_graph import SparseGraph
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import List
from utils import load_edges
from tqdm import tqdm

from walker import BiasedRandomWalker


# %%
DATASET_PATH = '../data/lab2_edge.csv'
graph = SparseGraph.build_graph_from_csv(DATASET_PATH)
edges = load_edges(DATASET_PATH)


# %%
N_EPOCHS = 75
N_WALKS_PER_BATCH = 5
WALK_LENGTH = 15
N_NODES = graph.coo.shape[0]
IN_DIM = N_NODES
HIDDEN_DIM = 2048
EMBEDDING_DIM = 64
WINDOW_SIZE = 5
BATCH_SIZE = 128
N_NEG_SAMPLES = 12
EPSILON = 1e-7
DEVICE = 'cuda:0'

walker = BiasedRandomWalker(graph, 1, 1)
model = SkipGram(IN_DIM, EMBEDDING_DIM).to(DEVICE)
model.train()
optimizer = optim.SGD(model.parameters(), lr=10, momentum=0.9)
loss_metric = NegativeSamplingLoss(N_NODES, N_NEG_SAMPLES)

# %%
for e in range(N_EPOCHS):
    walk_data: List[List[int]] = []
    isolated_nodes = set([])
    nodes = list(range(N_NODES))
    random.shuffle(nodes)
    for node in nodes:
        if graph.get_degree(node) == 0:
            isolated_nodes.add(node)
            continue
        walk = walker(node, WALK_LENGTH)
        for centroid in range(WINDOW_SIZE, WALK_LENGTH - WINDOW_SIZE):
            walk_data.append(
                walk[centroid - WINDOW_SIZE: centroid + WINDOW_SIZE + 1])
    walk_data = torch.LongTensor(walk_data)

    data_loader = DataLoader(walk_data, batch_size=BATCH_SIZE, shuffle=True)

    tot_loss = 0
    t = tqdm(data_loader)
    model.train()
    for bidx, x in enumerate(t):
        current_bsize = x.shape[0]
        optimizer.zero_grad()
        x = x.to(DEVICE)
        preds = model.forward(x)
        loss = loss_metric(model, preds, current_bsize)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        t.set_description(f'Epoch: {e:2d} Loss: {tot_loss / (bidx+1):.4f}')
    model.eval()

    neg_tests = get_negative_tests(graph, 2000)
    pos_tests = torch.tensor(
        edges[random.choices(list(range(edges.shape[0])), k=2000)])
    neg_tests = neg_tests.to(DEVICE)
    pos_tests = pos_tests.to(DEVICE)

    auc = calc_auc(neg_tests, pos_tests, model)

    print(f'Epoch: {e:2d} AUC: {auc:.4f}', file=sys.stderr)

# %%
SAVE_PATH = './pretrained.pt'
torch.save(model.state_dict(), SAVE_PATH)

# %%
import pandas as pd
TEST_PATH = '../data/lab2_test.csv'
test_set = pd.read_csv(TEST_PATH).to_numpy()
# %%
