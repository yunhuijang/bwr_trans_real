from rdkit import Chem
from rdkit.Chem import BRICS
from joblib import Parallel, delayed
from rdkit.Chem.rdmolfiles import MolToSmiles, MolFromSmiles
from brics_generator import BRICSGenerator, BRICSDecompose
import re


def filter_brics_fragments(dataset):
    '''
    to translate unique_fragment into unique_filtered_fragment
    replace [*number] to [*] and remove redundants
    '''

    with open(f'../resource/fragment/{dataset}/unique_fragment_list.txt', 'r') as f:
        fragment_smiles_list = [line.rstrip() for line in f]

    filtered_fragment_list = [re.subn("[[]\d[*][]]", '[*]', frag)[0] for frag in fragment_smiles_list]
    filtered_fraagment_list = [re.subn("[[]\d\d[*][]]", '[*]', frag)[0] for frag in filtered_fragment_list]
    filtered_fragment_set = set(filtered_fraagment_list)


    with open(f'../resource/fragment/{dataset}/unique_filtered_fragment_list.txt', 'w') as f:
        for smiles in filtered_fragment_set:
            f.write(f'{smiles}\n')
            

def reverse_fragment(fragment):
    # TODO: deal with reversing ions (ex: Cl -> lC)
    '''
    reverse fragments to deal with symmetric fragments
    ex) [*]CNCC -> CCNC[*]
    '''
    reversed_fragment = fragment[::-1]
    reversed_fragment = reversed_fragment.replace('(', '%temp%').replace(')', '(').replace('%temp%', ')')
    reversed_fragment = reversed_fragment.replace('[', '%temp%').replace(']', '[').replace('%temp%', ']')
    return reversed_fragment

def map_fragment_type(fragment_list):
    '''
    map fragment types to each fragment
    1) [*]CCN[*] -> reversed: 11 ...
    2) [*]CCCN
    3) [*]CC([*])N
    4) [*]CC([*])NC[*]
    '''
    fragment_type_dict = {}
    for frag in fragment_list:
        if frag.count('[*]') == 1:
            if frag[0] == '[':
                # type 2
                fragment_type_dict[frag] = 2
                fragment_type_dict[reverse_fragment(frag)] = 22
            else:
                raise ValueError(f"[*] not in the first position: {frag}")
        elif frag.count('[*]') > 2:
            # type 4
            fragment_type_dict[frag] = 4
            fragment_type_dict[reverse_fragment(frag)] = 44
        else:
            # type 1
            if frag[-3:] == '[*]':
                fragment_type_dict[frag] = 1
                fragment_type_dict[reverse_fragment(frag)] = 11
            else:
                # type 3
                fragment_type_dict[frag] = 3
                fragment_type_dict[reverse_fragment(frag)] = 33
                
    return fragment_type_dict

dataset = 'qm9'

with open(f'../resource/fragment/{dataset}/unique_filtered_fragment_list.txt', 'r') as f:
    org_fragment_list = [line.rstrip() for line in f]
fragment_type_dict = map_fragment_type(org_fragment_list)
fragment_list = list(fragment_type_dict.keys())

# tokenize
# smiles = 'CCCOCCC'
smiles = 'CCCOCCC(=O)c1ccccc1'
mol = Chem.MolFromSmiles(smiles)
gen_fragments = BRICSDecompose(mol, is_freq=True)
fragments = dict(filter(lambda elem: elem[1]>0, gen_fragments.items()))
fragments_with_type = {frag: {'freq': freq, 'type': fragment_type_dict[frag]} for frag, freq in fragments}


# # m = Chem.MolFromSmiles('CCCOCCC(=O)c1ccccc1')
# m = Chem.MolFromSmiles('CCCOCCC')
# # res = list(BRICSDecompose(m, singlePass=True))
# non_single = BRICSDecompose(m, is_freq=True)
# # print(res)
# print(non_single)