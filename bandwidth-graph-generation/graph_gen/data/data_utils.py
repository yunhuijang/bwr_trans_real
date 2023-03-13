import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import networkx as nx
from networkx import from_numpy_matrix, convert_node_labels_to_integers


def train_val_test_split(
    data: list,
    train_size: float = 0.7, val_size: float = 0.1, test_size: float = 0.2,
    seed: int = 42,
):
    train_val, test = train_test_split(data, train_size=train_size + test_size, random_state=seed)
    train, val = train_test_split(train_val, train_size=train_size / (train_size + val_size), random_state=seed)
    return train, val, test


def data_to_edge_set(data: Data):
    return set(
        tuple(data.edge_index[:, i].tolist())
        for i in range(data.edge_index.shape[1])
    )


def add_edge_set_to_data(
    data: Data, edge_set: set, edge_type: int,
) -> Data:
    # get edges in nx graph but not edege_set
    old_edge_set = data_to_edge_set(data)
    edges_to_add = edge_set - old_edge_set
    # add the backward edges
    edges_to_add = edges_to_add.union((j, i) for i, j in edges_to_add)
    # convert the new edges to an edge_index Tensor
    new_edge_index_tensor = torch.tensor(
        list(edges_to_add), dtype=torch.long,
    ).T
    new_edge_index_tensor = torch.cat([
        data.edge_index, new_edge_index_tensor,
    ], dim=-1)
    # build new edge type Tensor
    old_edge_type = (
        data.edge_type if hasattr(data, "edge_type")
        else torch.zeros(len(old_edge_set), dtype=torch.int)
    )
    new_edge_type = torch.full((len(edges_to_add),), fill_value=edge_type, dtype=torch.int)
    edge_type_tensor = torch.cat([old_edge_type, new_edge_type])
    # combine it all into a new Data object
    n = data.num_nodes
    data = data.clone()
    data.num_nodes = n
    data.edge_index = new_edge_index_tensor
    data.edge_type = edge_type_tensor
    new_E = new_edge_index_tensor.shape[1]
    if data.edge_attr is not None:
        data.edge_attr = torch.cat([
            data.edge_attr, torch.zeros((new_E, data.edge_attr.shape[1])),
        ])
    return data


def add_complete_edge_set(data: Data, edge_type: int) -> Data:
    n = data.num_nodes
    graph = nx.complete_graph(n=n)
    edge_set = set(graph.edges())
    return add_edge_set_to_data(data, edge_set, edge_type)


def add_nx_graph_to_data(
    data: Data, graph: nx.Graph, fill_val: float = 0,
) -> Data:
    # get edges in nx graph but not Data
    new_edge_set = set(graph.edges())
    edge_set = data_to_edge_set(data)
    edges_to_add = new_edge_set - edge_set
    # add the backward edges
    edges_to_add = edges_to_add.union((j, i) for i, j in edges_to_add)
    # convert the new edges to an edge_index Tensor
    new_edge_index_tensor = torch.tensor(
        list(edges_to_add), dtype=torch.long,
    ).T
    new_edge_index_tensor = torch.cat([
        data.edge_index, new_edge_index_tensor,
    ], dim=-1)
    # build new edge_attributes
    new_edge_attr = torch.cat([
        data.edge_attr,
        torch.full((len(edges_to_add), *data.edge_attr.shape[1:]), fill_val),
    ])
    # build edge type Tensor
    edge_type = torch.zeros(new_edge_attr.shape[0], dtype=torch.int)
    edge_type[data.edge_attr.shape[0]:] = 1
    # combine it all into a new Data object
    data = data.clone()
    data.edge_index = new_edge_index_tensor
    data.edge_type = edge_type
    data.edge_attr = new_edge_attr
    return data


def add_compete_graph(
    data: Data, fill_val: float = 0,
) -> Data:
    n = data.num_nodes
    graph = nx.complete_graph(n)
    return add_nx_graph_to_data(data, graph, fill_val)


def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def clean_graph_from_adj(adj: np.ndarray) -> nx.Graph:
    G = nx.from_numpy_array(adj)
    G.remove_nodes_from(list(nx.isolates(G))),
    G = nx.convert_node_labels_to_integers(G)
    return G

def sequence_to_bw_matrix(sequence, token_to_id, vocab_size):
    '''
    sequence of tokens to adjacency matrix
    sequence: sequence without start / end token
    '''
    id_to_token = {value: key for key, value in token_to_id.items()}
    # 0 values for padding token and end token
    id_to_token[0] = tuple(np.zeros(len(list(token_to_id.keys())[0])))
    id_to_token[vocab_size-1] = tuple(np.zeros(len(list(token_to_id.keys())[0])))
    bw_matrix = [list(id_to_token[token]) for token in sequence]
    
    return bw_matrix

def bw_matrix_to_adj(bw_matrix):
    '''
    bandwidth matrix (N*bw) to adjacency matrix
    '''
    # remove first column (start / end token)
    band_flat_A = torch.tensor(bw_matrix).T[1:].T
    n, bw = band_flat_A.shape[:2]
    out = torch.zeros((n, n) + band_flat_A.shape[2:], dtype=band_flat_A.dtype, device=band_flat_A.device)
    for i in range(n):
        append_len = min(bw, n - i - 1)
        if append_len > 0:
            out[i, i + 1: i + 1 + append_len] = band_flat_A[i, :append_len]
    adj = out + out.T
    return adj
    
def adj_to_graph(adj):
    '''
    adjacency matrix to graph
    '''
    graph = from_numpy_matrix(adj.cpu().numpy())
    graph.remove_nodes_from(list(nx.isolates(graph)))
    try:
        graph = graph.subgraph(max(nx.connected_components(graph), key=len))
        graph = convert_node_labels_to_integers(graph)
    except:
        # no graph (sequence of 0s)
        graph = None
    return graph


