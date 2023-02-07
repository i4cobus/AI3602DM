from collections import Counter
import sys
import random
import typing as t
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm import trange, tqdm
from utility import *

random.seed("HAPPY4EVER")


def rebuild_metagraph_coo_mat(old_graph: Graph, n_nodes: int) -> sparse.coo_matrix:
    reindexer = graph_reindex(old_graph)
    new_src = []
    new_dst = []
    edge_weights = []
    for node in range(n_nodes):
        comm_src = metagraph.node2comm[node]
        comm_src_ridx = reindexer[comm_src]
        for neighbour in metagraph.get_out_neighbours(node):
            comm_dst = metagraph.node2comm[neighbour]
            comm_dst_ridx = reindexer[comm_dst]
            new_src.append(comm_src_ridx)
            new_dst.append(comm_dst_ridx)
            edge_weights.append(metagraph.edge_weights[(node, neighbour)])
    new_coo_matrix = sparse.coo_matrix((edge_weights, (new_src, new_dst)))
    new_coo_matrix.sum_duplicates()

    return new_coo_matrix

def merge_community(idx1: int, idx2: int, communities: t.List[Community]):
    comm1 = communities[idx1]
    comm2 = communities[idx2]
    for node in comm2.nodes.copy():
        comm1.add_node(node)
        comm2.remove_node(node)
        global_node2comm[node] = idx1

metagraph = Graph(graph_coo)
GRAPH = Graph(graph_coo)
global_node2comm: t.List[int] = []
node2comm: t.List[int] = []
communities: t.List[Community] = []

for node in range(n_nodes):
    global_node2comm.append(node)
    node2comm.append(node)
    community = Community(metagraph)
    community.add_node(node)
    communities.append(community)

metagraph.set_node_to_comm(node2comm)

num_iter = 0
modularity = 0
while True:
    while True:
        num_iter += 1
        print("Iteration", num_iter, file=sys.stderr)
        changed = False
        num_nodes_changed = 0
        random_iterator = list(range(n_nodes))
        random.shuffle(random_iterator)
        for node in tqdm(random_iterator):
            old_comm_idx = metagraph.node2comm[node]
            old_comm = communities[old_comm_idx]
            old_comm.remove_node(node)

            best_comm_idx = old_comm_idx
            best_comm = old_comm

            metagraph.node2comm[node] = -1
            best_modularity = 0

            delta_q_rm = nodewise_delta_q(node, old_comm)

            for neighbour in metagraph.get_neighbours(node):
                new_comm_idx = metagraph.node2comm[neighbour]
                if new_comm_idx == old_comm_idx:
                    continue
                new_comm = communities[new_comm_idx]
                delta_q_mv = nodewise_delta_q(node, new_comm)
                d_q = delta_q_mv - delta_q_rm
                if d_q > best_modularity:
                    changed = True
                    best_modularity = d_q
                    best_comm = new_comm
                    best_comm_idx = new_comm_idx

            metagraph.node2comm[node] = best_comm_idx
            modularity += best_modularity

            best_comm.add_node(node)
            if old_comm_idx != best_comm_idx:
                num_nodes_changed += 1

        print(
            f'Modularity: {modularity:.4f} | '
            + f'Changed: {num_nodes_changed} | '
            + f'Communities: {len(set(metagraph.node2comm))}', file=sys.stderr)
        if not changed:
            break


    reindexer = graph_reindex(metagraph)

    for node in range(N_NODES):
        metanode = global_node2comm[node]  
        new_metanode = metagraph.node2comm[metanode]  
        new_metanode_ridx = reindexer[new_metanode]  
        global_node2comm[node] = new_metanode_ridx
    GRAPH.set_node_to_comm(global_node2comm)

    new_coo_matrix = rebuild_metagraph_coo_mat(metagraph, n_nodes)

    if graph_coo.shape == new_coo_matrix.shape:
        if (graph_coo - new_coo_matrix).sum() == 0:
            print('There is no change in metagraph structure. \n Done!!')
            break

    graph_coo = new_coo_matrix
    metagraph = Graph(graph_coo)
    node2comm: t.List[int] = []
    communities: t.List[Community] = []

    n_nodes = graph_coo.shape[0]

    for node in range(n_nodes):
        node2comm.append(node)
        community = Community(metagraph)
        community.add_node(node)
        communities.append(community)

    metagraph.set_node_to_comm(node2comm)

community_counter = Counter(global_node2comm)
communities = []

for i in range(len(community_counter)):
    communities.append(Community(GRAPH))

for node in range(N_NODES):
    comm = global_node2comm[node]
    communities[comm].add_node(node)

while len(community_counter) > 5:
    print(len(community_counter))
    sorted_counter = sorted(community_counter.items(), key=lambda x: x[1])
    candidates = sorted_counter[:2]
    best_dq = -100
    best_pair: t.Tuple[int, int] = (candidates[0][0], candidates[1][0])
    if best_pair[0] == -1 or best_pair[1] == -1:
        break
    merge_community(candidates[0][0], candidates[1][0], communities)
    community_counter = Counter(global_node2comm)

with open('./res.csv', 'w') as fo:
    fo.write('id, category\n')
    reindexer = graph_reindex(GRAPH)
    for id, cat in enumerate(global_node2comm):
        fo.write(f'{id}, {reindexer[cat]}\n')
