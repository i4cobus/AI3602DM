from collections import Counter
import typing as t
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from collections import defaultdict as ddict

class Graph():
    def __init__(self,
                 adj_mat: sparse.coo_matrix,):
        self.coo_matrix = adj_mat
        self.out_degrees = self.coo_matrix.sum(axis=1).A1
        self.in_degrees = self.coo_matrix.sum(axis=0).A1
        self.neighbours: ddict[int, t.Set[int]] = ddict(set)
        self.in_neighbours: ddict[int, t.Set[int]] = ddict(set)
        self.out_neighbours: ddict[int, t.Set[int]] = ddict(set)
        self.edge_weights: t.Dict[t.Tuple[int, int], int] = {}
        for src, dst, weight in zip(self.coo_matrix.row, self.coo_matrix.col, self.coo_matrix.data):
            self.neighbours[src].add(dst)
            self.neighbours[dst].add(src)
            self.in_neighbours[dst].add(src)
            self.out_neighbours[src].add(dst)
            self.edge_weights[(src, dst)] = weight
        self.M = float(self.coo_matrix.sum())

    def set_node_to_comm(self, node2comm: t.List[int]):
        self.node2comm = node2comm

    def get_in_degrees(self, node: int) -> int:
        return self.in_degrees[node]

    def get_out_degrees(self, node: int) -> int:
        return self.out_degrees[node]

    def get_neighbours(self, node: int) -> t.Set[int]:
        return self.neighbours[node]

    def get_in_neighbours(self, node: int) -> t.Set[int]:
        return self.in_neighbours[node]

    def get_out_neighbours(self, node: int) -> t.Set[int]:
        return self.out_neighbours[node]

class Community():
    def __init__(self, graph: Graph):
        self.nodes: t.Set[int] = set([])
        self.G = graph
        self.out_degree: int = 0
        self.in_degree: int = 0

    def add_node(self, node: int):
        self.nodes.add(node)
        self.out_degree += self.G.out_degrees[node]
        self.in_degree += self.G.in_degrees[node]

    def remove_node(self, node: int):
        self.nodes.remove(node)
        self.out_degree -= self.G.out_degrees[node]
        self.in_degree -= self.G.in_degrees[node]

    def intra_comm_in_degree(self, node: int) -> float:
        in_degree = 0.
        in_neighbours = self.G.get_in_neighbours(node)
        for neighbour in in_neighbours:
            if neighbour in self.nodes:
                in_degree += self.G.edge_weights[(neighbour, node)]

        return in_degree

    def intra_comm_out_degree(self, node: int) -> float:
        out_degree = 0.
        out_neighbours = self.G.get_out_neighbours(node)
        for neighbour in out_neighbours:
            if neighbour in self.nodes:
                out_degree += self.G.edge_weights[(node, neighbour)]

        return out_degree

N_NODES = 31136
n_nodes = N_NODES
EDGE_CSV_PATH = './data/edges_update.csv'

edges = pd.read_csv(EDGE_CSV_PATH).to_numpy()

graph_coo = sparse.coo_matrix(
    (np.ones(edges.shape[0]), (edges.T[0], edges.T[1])),
)

graph_coo.sum_duplicates()

M: int = graph_coo.sum()

def nodewise_delta_q(node: int, community: Community):
    graph = community.G
    intra_in_degree = float(community.intra_comm_in_degree(node))
    intra_out_degree = float(community.intra_comm_out_degree(node))
    in_degree = float(graph.get_in_degrees(node))
    out_degree = float(graph.get_out_degrees(node))
    comm_out_degree = float(community.out_degree)
    comm_in_degree = float(community.in_degree)

    return (intra_in_degree + intra_out_degree) / graph.M -\
        (in_degree * comm_out_degree + out_degree * comm_in_degree) / (graph.M ** 2)

def commwise_delta_q(comm1: Community, comm2: Community):
    dq_loss = (comm1.in_degree * comm2.out_degree + comm1.out_degree * comm2.in_degree)
    dq_loss = dq_loss / M / M
    dq_gain = 0
    for node in comm2.nodes:
        dq_gain += comm1.intra_comm_out_degree(node)
        dq_gain += comm1.intra_comm_in_degree(node)

    dq_gain = dq_gain / M

    return dq_gain - dq_loss

def graph_reindex(graph: Graph) -> t.Dict[int, int]:
    reindexer: t.Dict[int, int] = {}
    comms = set(graph.node2comm)
    for idx, comm in enumerate(comms):
        reindexer[comm] = idx

    return reindexer


def refresh_communities(
        global_node2comm: t.List[int],
        GRAPH: Graph):
    community_counter = Counter(global_node2comm)
    communities: t.List[Community] = []
    for i in range(len(community_counter)):
        communities.append(Community(GRAPH))
    for node in range(N_NODES):
        comm = global_node2comm[node]
        communities[comm].add_node(node)

    return communities



