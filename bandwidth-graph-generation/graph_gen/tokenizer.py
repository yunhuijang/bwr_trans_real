import pickle
import torch

from graph_gen.data import DATASETS
from graph_gen.data.orderings import order_graphs, ORDER_FUNCS
from graph_gen.data.nx_dataset import LSTMDataset


def generate_tokenizers(data_name, order):
    graph_getter, num_rep = DATASETS[data_name]
    if data_name == 'zinc250k':
        graphs = graph_getter(zinc_path='resource/zinc.csv')
    elif data_name == 'peptides':
        graphs = graph_getter(peptide_path='')
    else:
        graphs = graph_getter()
    
    # map order graphs
    order_func = ORDER_FUNCS[order]
    ordered_graphs = order_graphs(graphs, num_repetitions=num_rep, order_func=order_func, seed=0)
    
    # set bw
    bw = max([graph.bw for graph in ordered_graphs])
    print(bw)
    # TODO: fix circular in LSTMDataset
    total_dataset = LSTMDataset(ordered_graphs, bw, data_name)
    
    inp_list = [inp for inp, out in total_dataset.band_flat_adjs]
    token_set = set(tuple(a.numpy()) for a in set().union(*[torch.unique(inp, dim=0) for inp in inp_list]))
    sorted_token_list = sorted(list(token_set), reverse=True)
    token_to_id_dict = {token: id+1 for id, token in enumerate(sorted_token_list)}
    # make empty token
    # token_to_id_dict[tuple(np.zeros(len(list(token_to_id_dict.keys())[0])))] = 0
    with open(f'resource/tokenizer/{data_name}.pkl', 'wb') as f:
        pickle.dump(token_to_id_dict, f)
        
data = 'community2'
generate_tokenizers(data, 'C-M')