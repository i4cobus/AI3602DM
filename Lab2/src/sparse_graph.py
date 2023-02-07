import numpy as np
import scipy.sparse as sparse
from collections import defaultdict as ddict
from utils import load_edges
from typing import List


class SparseGraph():
    def __init__(
            self,
            coo: sparse.coo_matrix):
        self.coo = coo
        self.degrees = np.zeros(self.coo.shape[0])
        self.neighbours: ddict[int, List[int]] = ddict(list)
        for src, dst in zip(self.coo.row, self.coo.col):
            if dst not in self.neighbours[src]:
                self.neighbours[src].append(dst)
                self.degrees[src] += 1
            if src not in self.neighbours[dst]:
                self.neighbours[dst].append(src)
                self.degrees[dst] += 1

    def get_neighbours(self, node: int) -> List[int]:
        return self.neighbours[node]

    def get_degree(self, node: int) -> int:
        return self.degrees[node]

    @classmethod
    def build_graph_from_csv(cls, path: str):
        edges = load_edges(path).T
        srcs, dsts = edges[0], edges[1]
        coo = sparse.coo_matrix(
            (np.ones(edges.shape[1]), (srcs, dsts)))
        coo.sum_duplicates()
        return cls(coo)
