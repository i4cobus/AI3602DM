import random

from typing import Dict, List
from abc import abstractmethod
from sparse_graph import SparseGraph


class RandomWalker():
    def __init__(self, g: SparseGraph):
        self.G = g

    def _uniform_sampler(self, current):
        return random.choice(self.G.get_neighbours(current))

    @abstractmethod
    def __call__(self, start: int, length: int) -> List[int]:
        return []


class UniformRandomWalker(RandomWalker):
    def __init__(self, g: SparseGraph):
        super().__init__(g)

    def __call__(self, start: int, length: int) -> List[int]:
        trace = []
        trace.append(start)
        current_len = 1
        current = start
        while current_len < length:
            target = self._uniform_sampler(current)
            trace.append(target)
            current = target
            current_len += 1

        return trace


class BiasedRandomWalker(RandomWalker):
    def __init__(self, g: SparseGraph, ret_param=1, inout_param=1):
        super().__init__(g)
        self.ret_p = ret_param
        self.io_q = inout_param
        self.transition_table: Dict[Dict[List]] = {}
        self._build_transition_table()

    def _unnormalized_prob(self, his, nxt):
        if his == nxt:
            return 1 / self.ret_p
        if nxt in self.G.get_neighbours(his):
            return 1
        else:
            return 1 / self.io_q

    def _normalize(self, probs):
        factor = sum(probs)
        return [p / factor for p in probs]

    def _build_transition_table(self):
        N_NODES = self.G.coo.shape[0]
        transition_table = {}
        for his in range(N_NODES):
            hist_neighbours = self.G.get_neighbours(his)
            if len(hist_neighbours) == 0:
                continue
            cur_dict = {}
            for cur in hist_neighbours:
                cur_neighbours = self.G.get_neighbours(cur)
                if len(cur_neighbours) == 0:
                    continue
                cur2next = [0] * len(cur_neighbours)
                for idx, nxt in enumerate(cur_neighbours):
                    cur2next[idx] = self._unnormalized_prob(his, nxt)
                cur2next = self._normalize(cur2next)
                cur_dict[cur] = cur2next
            transition_table[his] = cur_dict
        self.transition_table = transition_table

    def __call__(self, start: int, length: int) -> List[int]:
        trace = []
        trace.append(start)
        current_len = 1
        current = start
        while current_len < length:
            if current_len == 1:
                target = self._uniform_sampler(current)
            else:
                hist = trace[current_len - 2]
                trans_prob = self.transition_table[hist][current]
                target = random.choices(
                    self.G.get_neighbours(current),
                    trans_prob,
                    k=1)[0]
            trace.append(target)
            current = target
            current_len += 1

        return trace
