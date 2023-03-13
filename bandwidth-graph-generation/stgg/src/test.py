from rdkit import Chem
from rdkit.Chem import BRICS
import random
import copy
import os
import re
import random
from random import sample
import argparse
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem.rdmolfiles import MolToSmiles, MolFromSmiles
from joblib import Parallel, delayed
import wandb
import torch
from moses.metrics.metrics import get_all_metrics
from tqdm import tqdm
from collections import defaultdict

from data.smiles_dataset import ZincDataset, MosesDataset, SimpleMosesDataset, QM9Dataset, untokenize
from GDSS.utils.mol_utils import smiles_to_mols, mols_to_nx
from GDSS.evaluation.mmd import compute_nspdk_mmd
from util import canonicalize

def str2bool(v):
  if isinstance(v, bool):
      return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
  else:
      raise argparse.ArgumentTypeError('Boolean value expected.')

environs = {
  'L1': '[C;D3]([#0,#6,#7,#8])(=O)',
  #
  # After some discussion, the L2 definitions ("N.pl3" in the original
  # paper) have been removed and incorporated into a (almost) general
  # purpose amine definition in L5 ("N.sp3" in the paper).
  #
  # The problem is one of consistency.
  #    Based on the original definitions you should get the following
  #    fragmentations:
  #      C1CCCCC1NC(=O)C -> C1CCCCC1N[2*].[1*]C(=O)C
  #      c1ccccc1NC(=O)C -> c1ccccc1[16*].[2*]N[2*].[1*]C(=O)C
  #    This difference just didn't make sense to us. By switching to
  #    the unified definition we end up with:
  #      C1CCCCC1NC(=O)C -> C1CCCCC1[15*].[5*]N[5*].[1*]C(=O)C
  #      c1ccccc1NC(=O)C -> c1ccccc1[16*].[5*]N[5*].[1*]C(=O)C
  #
  # 'L2':'[N;!R;!D1;!$(N=*)]-;!@[#0,#6]',
  # this one turned out to be too tricky to define above, so we set it off
  # in its own definition:
  # 'L2a':'[N;D3;R;$(N(@[C;!$(C=*)])@[C;!$(C=*)])]',
  'L3': '[O;D2]-;!@[#0,#6,#1]',
  'L4': '[C;!D1;!$(C=*)]-;!@[#6]',
  # 'L5':'[N;!D1;!$(N*!-*);!$(N=*);!$(N-[!C;!#0])]-[#0,C]',
  'L5': '[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]',
  'L6': '[C;D3;!R](=O)-;!@[#0,#6,#7,#8]',
  'L7a': '[C;D2,D3]-[#6]',
  'L7b': '[C;D2,D3]-[#6]',
  '#L8': '[C;!R;!D1]-;!@[#6]',
  'L8': '[C;!R;!D1;!$(C!-*)]',
  'L9': '[n;+0;$(n(:[c,n,o,s]):[c,n,o,s])]',
  'L10': '[N;R;$(N(@C(=O))@[C,N,O,S])]',
  'L11': '[S;D2](-;!@[#0,#6])',
  'L12': '[S;D4]([#6,#0])(=O)(=O)',
  'L13': '[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]',
  'L14': '[c;$(c(:[c,n,o,s]):[n,o,s])]',
  'L14b': '[c;$(c(:[c,n,o,s]):[n,o,s])]',
  'L15': '[C;$(C(-;@C)-;@C)]',
  'L16': '[c;$(c(:c):c)]',
  'L16b': '[c;$(c(:c):c)]',
}

dummyPattern = Chem.MolFromSmiles('[*]')
reactionDefs = (
  # L1
  [
    ('1', '3', '-'),
    ('1', '5', '-'),
    ('1', '10', '-'),
  ],

  # L3
  [
    ('3', '4', '-'),
    ('3', '13', '-'),
    ('3', '14', '-'),
    ('3', '15', '-'),
    ('3', '16', '-'),
  ],

  # L4
  [
    ('4', '5', '-'),
    ('4', '11', '-'),
  ],

  # L5
  [
    ('5', '12', '-'),
    ('5', '14', '-'),
    ('5', '16', '-'),
    ('5', '13', '-'),
    ('5', '15', '-'),
  ],

  # L6
  [
    ('6', '13', '-'),
    ('6', '14', '-'),
    ('6', '15', '-'),
    ('6', '16', '-'),
  ],

  # L7
  [
    ('7a', '7b', '='),
  ],

  # L8
  [
    ('8', '9', '-'),
    ('8', '10', '-'),
    ('8', '13', '-'),
    ('8', '14', '-'),
    ('8', '15', '-'),
    ('8', '16', '-'),
  ],

  # L9
  [
    ('9', '13', '-'),  # not in original paper
    ('9', '14', '-'),  # not in original paper
    ('9', '15', '-'),
    ('9', '16', '-'),
  ],

  # L10
  [
    ('10', '13', '-'),
    ('10', '14', '-'),
    ('10', '15', '-'),
    ('10', '16', '-'),
  ],

  # L11
  [
    ('11', '13', '-'),
    ('11', '14', '-'),
    ('11', '15', '-'),
    ('11', '16', '-'),
  ],

  # L12
  # none left

  # L13
  [
    ('13', '14', '-'),
    ('13', '15', '-'),
    ('13', '16', '-'),
  ],

  # L14
  [
    ('14', '14', '-'),  # not in original paper
    ('14', '15', '-'),
    ('14', '16', '-'),
  ],

  # L15
  [
    ('15', '16', '-'),
  ],

  # L16
  [
    ('16', '16', '-'),  # not in original paper
  ], )
smartsGps = copy.deepcopy(reactionDefs)
# smartsGps: set of fragment rules
# Changes each fragment rules into SMARTS reaction
for gp in smartsGps:
    # defn: one reaction in each fragment rule
    for j, defn in enumerate(gp):
        g1, g2, bnd = defn
        r1 = environs['L' + g1]
        r2 = environs['L' + g2]
        g1 = re.sub('[a-z,A-Z]', '', g1)
        g2 = re.sub('[a-z,A-Z]', '', g2)
        sma = f'[$({r1}):1]{bnd};!@[$({r2}):2]>>[{g1}*]-[*:1].[{g2}*]-[*:2]'
        gp[j] = sma

# Maps each SMARTS reaction into reaction object
for gp in smartsGps:
    for defn in gp:
        try:
            t = Reactions.ReactionFromSmarts(defn)
            t.Initialize()
        except Exception:
            print(defn)
            raise

reactions = tuple([[Reactions.ReactionFromSmarts(y) for y in x] for x in smartsGps])

environMatchers = {}
for env, sma in environs.items():
    environMatchers[env] = Chem.MolFromSmarts(sma)

bondMatchers = []
for i, compats in enumerate(reactionDefs):
    tmp = []
    for i1, i2, bType in compats:
        e1 = environs['L%s' % i1]
        e2 = environs['L%s' % i2]
        patt = '[$(%s)]%s;!@[$(%s)]' % (e1, bType, e2)
        patt = Chem.MolFromSmarts(patt)
        tmp.append((i1, i2, bType, patt))
    bondMatchers.append(tmp)

# From product to reactant
reverseReactions = []
for i, rxnSet in enumerate(smartsGps):
    for j, sma in enumerate(rxnSet):
        rs, ps = sma.split('>>')
        sma = '%s>>%s' % (ps, rs)
        rxn = Reactions.ReactionFromSmarts(sma)
        labels = re.findall(r'\[([0-9]+?)\*\]', ps)
        rxn._matchers = [Chem.MolFromSmiles('[%s*]' % x) for x in labels]
        reverseReactions.append(rxn)


mol = MolFromSmiles('CCCOCCC(=O)c1ccccc1')
randomizeOrder = False
silent = True
letter = re.compile('[a-z,A-Z]')
indices = list(range(len(bondMatchers)))
bondsDone = set()
if randomizeOrder:
    random.shuffle(indices, random=random.random)

envMatches = {}
# find environments that has substructure in the molecule
for env, patt in environMatchers.items():
    envMatches[env] = mol.HasSubstructMatch(patt)
for gpIdx in indices:
    if randomizeOrder:
        compats = bondMatchers[gpIdx][:]
        random.shuffle(compats, random=random.random)
    else:
        # compats: 각 attachment label에 따른 reaction 
        compats = bondMatchers[gpIdx]
    for i1, i2, bType, patt in compats:
        if not envMatches['L' + i1] or not envMatches['L' + i2]:
            continue
        # matches: index of bond in SMILES
        matches = mol.GetSubstructMatches(patt)
        i1 = letter.sub('', i1)
        i2 = letter.sub('', i2)
        for match in matches:
            if match not in bondsDone and (match[1], match[0]) not in bondsDone:
                bondsDone.add(match)
                print((match[0], match[1], (i1, i2)))
                # yield (((match[0], match[1]), (i1, i2)))