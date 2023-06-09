import os
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm
from rdkit import Chem


DATASET_PATH = os.path.join(
    Path(os.path.dirname(os.path.realpath(__file__))).parent.parent,
    "datasets",
)


def smiles_to_nx_no_features(sm: str) -> nx.Graph:
    mol = Chem.MolFromSmiles(sm)
    adj = Chem.GetAdjacencyMatrix(mol)
    return nx.from_numpy_matrix(adj)


def get_zinc_graphs(
    zinc_path
):
    zinc = pd.read_csv(zinc_path)
    zinc_graphs = list(map(smiles_to_nx_no_features, zinc["smiles"].values))
    return zinc_graphs


def get_peptide_graphs(
    peptide_path
):
    smiles = pd.read_csv(peptide_path, usecols=["smiles"])["smiles"].values
    graphs = list(map(smiles_to_nx_no_features, tqdm(smiles)))
    graphs = [graph for graph in graphs if nx.number_connected_components(graph) == 1]
    print("initial length", len(smiles), "-> filtered length", len(graphs))
    return graphs

# TODO: graphs_to_smiles
def graph_to_smiles(graph):
    # stgg/data/target_data/to_smiles
    
    pass