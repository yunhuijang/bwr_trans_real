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

def BRICSDecompose(mol, allNodes=None, minFragmentSize=1, onlyUseReactions=None, silent=True,
                   keepNonLeafNodes=False, singlePass=False, returnMols=False, is_freq=False):
  global reactions
  mSmi = Chem.MolToSmiles(mol, 1)

  if allNodes is None:
      allNodes = set()

  if mSmi in allNodes:
      return set()

  activePool = {mSmi: mol}
  # activePool_dict: frequency of each fragments
  activePool_dict = defaultdict(lambda:0)
  allNodes.add(mSmi)
  foundMols = {mSmi: mol}
  for gpIdx, reactionGp in enumerate(reactions):
      newPool = {}
      while activePool:
          matched = False
          nSmi = next(iter(activePool))
          mol = activePool.pop(nSmi)
          for rxnIdx, reaction in enumerate(reactionGp):
              if onlyUseReactions and (gpIdx, rxnIdx) not in onlyUseReactions:
                  continue
              # if not silent:
              #     print('--------')
              #     print(smartsGps[gpIdx][rxnIdx])
              ps = reaction.RunReactants((mol, ))
              # ps: all possible reactions
              if ps:
                  # filter out redundant reactions
                  ps_set = set()
                  ps_list = []
                  for prodSeq in ps:
                    prod1, prod2 = prodSeq
                    ps_set.add((MolToSmiles(prod1), MolToSmiles(prod2)))
                  for prod1, prod2 in ps_set:
                    ps_list.append((MolFromSmiles(prod1), MolFromSmiles(prod2)))
                  
                  # nSmi is decopmosed into components of prodSeq
                  if not silent:
                    print(nSmi, '->', len(ps), 'products')
                    for prodSeq in ps_list:
                      prod1, prod2 = prodSeq
                      print((MolToSmiles(prod1), MolToSmiles(prod2)))
                  # filter out multiple reactions (apply only one reaction priotrized by the atoms size of smallest component)
                  if is_freq:
                    min_atom_size = 500
                    for prodSeq in ps_list:
                      prod1, prod2 = prodSeq
                      minimum_atomsize = min(prod1.GetNumAtoms(onlyExplicit=True), prod2.GetNumAtoms(onlyExplicit=True))
                      if minimum_atomsize < min_atom_size:
                        selected_ps = prodSeq
                      
                    ps_list = [prodSeq]
                  
                  for prodSeq in ps_list:
                      seqOk = True
                      # we want to disqualify small fragments, so sort the product sequence by size
                      tSeq = [(prod.GetNumAtoms(onlyExplicit=True), idx)
                              for idx, prod in enumerate(prodSeq)]
                      tSeq.sort()
                      # map pSmi (SMILES of prod) for each product
                      for nats, idx in tSeq:
                          prod = prodSeq[idx]
                          try:
                              Chem.SanitizeMol(prod)
                          except Exception:
                              continue
                          pSmi = Chem.MolToSmiles(prod, 1)
                          if minFragmentSize > 0:
                              nDummies = pSmi.count('*')
                              if nats - nDummies < minFragmentSize:
                                  seqOk = False
                                  break
                          prod.pSmi = pSmi
                      ts = [(x, prodSeq[y]) for x, y in tSeq]
                      prodSeq = ts
                      # add product molecules to activePool & allNodes
                      if seqOk:
                          matched = True
                          for nats, prod in prodSeq:
                              if nSmi in activePool_dict.keys() and activePool_dict[nSmi] > 0:
                                activePool_dict[nSmi] -= 1
                              else:
                                activePool_dict[nSmi] = 0
                              pSmi = prod.pSmi
                              if not singlePass:
                                  activePool[pSmi] = prod
                                  activePool_dict[pSmi] += 1
                              allNodes.add(pSmi)
                              foundMols[pSmi] = prod
          # current nSmi is not used -> need to be decomposed more
          if singlePass or keepNonLeafNodes or not matched:
              newPool[nSmi] = mol
      activePool = newPool
  if not (singlePass or keepNonLeafNodes):
      if not returnMols:
          res = set(activePool.keys())
      else:
          res = activePool.values()
  else:
      if not returnMols:
          res = allNodes
      else:
          res = foundMols.values()
  if is_freq:
    res = activePool_dict
  return res

def BRICSBuild(fragments, onlyCompleteMols=True, seeds=None, uniquify=True, scrambleReagents=True,
               maxDepth=3):
    seen = set()
    if not seeds:
        seeds = list(fragments)
    if scrambleReagents:
        seeds = list(seeds)
        random.shuffle(seeds, random=random.random)
    if scrambleReagents:
        tempReactions = list(reverseReactions)
        random.shuffle(tempReactions, random=random.random)
    else:
        tempReactions = reverseReactions
        
    # seeds: list of fragments
    for seed in seeds:
        seedIsR1 = False
        seedIsR2 = False
        nextSteps = []
        # rxn: one reaction
        for rxn in tempReactions:
            # seed가 reaction의 reactant / agent 중 무엇이랑 matching하는 지 check
            if seed.HasSubstructMatch(rxn._matchers[0]):
                seedIsR1 = True
            if seed.HasSubstructMatch(rxn._matchers[1]):
                seedIsR2 = True
            # fragment들 중 해당 reaction의 agent (seed=reactant) / reactant (seed=agent)가 될 수 있는 후보 만들기 -> product (ps) 생성
            for fragment in fragments:
                ps = None
                if fragment.HasSubstructMatch(rxn._matchers[0]):
                    if seedIsR2:
                        ps = rxn.RunReactants((fragment, seed))
                if fragment.HasSubstructMatch(rxn._matchers[1]):
                    if seedIsR1:
                        ps = rxn.RunReactants((seed, fragment))
                # ps: products
                if ps:
                    for p in ps:
                        if uniquify:
                            # pSmi: 해당 product (p)의 smiles
                            pSmi = Chem.MolToSmiles(p[0], True)
                            if pSmi in seen:
                                continue
                            else:
                                seen.add(pSmi)
                        # 만들어진 p에 dummy *가 있으면 이에 대해서 추가적으로 fragment 붙이기 필요
                        if p[0].HasSubstructMatch(dummyPattern):
                            nextSteps.append(p[0])
                            if not onlyCompleteMols:
                                yield p[0]
                        else:
                            yield p[0]
        if nextSteps and maxDepth > 0:
            for p in BRICSBuild(fragments, onlyCompleteMols=onlyCompleteMols, seeds=nextSteps,
                                uniquify=uniquify, maxDepth=maxDepth - 1,
                                scrambleReagents=scrambleReagents):
                if uniquify:
                    pSmi = Chem.MolToSmiles(p, True)
                    if pSmi in seen:
                        continue
                    else:
                        seen.add(pSmi)
                yield p


def FindBRICSBonds(mol, randomizeOrder=False, silent=True):
  # order: reaction order
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

m = MolFromSmiles('CCCOCCC(=O)c1ccccc1')
print(FindBRICSBonds(m, True))

def BreakBRICSBonds(mol, bonds=None, sanitize=True, silent=True):
    if not bonds:
        #bonds = FindBRICSBonds(mol)
        res = Chem.FragmentOnBRICSBonds(mol)
        if sanitize:
            Chem.SanitizeMol(res)
        return res
    eMol = Chem.EditableMol(mol)
    nAts = mol.GetNumAtoms()

    dummyPositions = []
    for indices, dummyTypes in bonds:
        ia, ib = indices
        obond = mol.GetBondBetweenAtoms(ia, ib)
        bondType = obond.GetBondType()
        eMol.RemoveBond(ia, ib)

        da, db = dummyTypes
        atoma = Chem.Atom(0)
        atoma.SetIsotope(int(da))
        atoma.SetNoImplicit(True)
        idxa = nAts
        nAts += 1
        eMol.AddAtom(atoma)
        eMol.AddBond(ia, idxa, bondType)

        atomb = Chem.Atom(0)
        atomb.SetIsotope(int(db))
        atomb.SetNoImplicit(True)
        idxb = nAts
        nAts += 1
        eMol.AddAtom(atomb)
        eMol.AddBond(ib, idxb, bondType)
        if mol.GetNumConformers():
            dummyPositions.append((idxa, ib))
            dummyPositions.append((idxb, ia))
    res = eMol.GetMol()
    if sanitize:
        Chem.SanitizeMol(res)
    if mol.GetNumConformers():
        for conf in mol.GetConformers():
            resConf = res.GetConformer(conf.GetId())
            for ia, pa in dummyPositions:
                resConf.SetAtomPosition(ia, conf.GetAtomPosition(pa))
    return res




# # get fragments from dataset

# # class BRICSGenerator():
#   # def __init__(self, hparams):
#   #     hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
#   #     self.setup_datasets(hparams)
#   #     self.device = torch.device("cuda:0")
    
#   # def setup_datasets(self, hparams):
#   #   dataset_cls = {
#   #       "zinc": ZincDataset,
#   #       "moses": MosesDataset,
#   #       "simplemoses": SimpleMosesDataset,
#   #       "qm9": QM9Dataset,
#   #   }.get(hparams.dataset_name)
#   #   self.train_dataset = dataset_cls("train_val", hparams.string_type)
#   #   self.val_dataset = dataset_cls("valid", hparams.string_type)
#   #   self.test_dataset = dataset_cls("test", hparams.string_type)
#   #   self.train_smiles_set = set(self.train_dataset.smiles_list)
  
#   # def check_samples(self, smiles_list):
#   #   train_data = self.train_dataset.smiles_list
#   #   test_data = self.test_dataset.smiles_list
#   #   valid_smiles_list = [smiles for smiles in smiles_list if smiles is not None]
#   #   wandb.init(name=f'{hparams.dataset_name}-brics-{hparams.random_seed_size}', project='benckmark', group=hparams.group)
#   #   wandb.config.update(hparams)
#   #   # calculate FCD
#   #   print("Comptue metrics")
#   #   if len(valid_smiles_list) > 0:
#   #       moses_statistics = get_all_metrics(
#   #           smiles_list, 
#   #           n_jobs=hparams.num_workers,
#   #           train=train_data,
#   #           device=str(self.device),
#   #           test=test_data,
#   #           batch_size=256
#         # )
#     print("Compute NSPDK")
#     # calculate NSPDK
#     test_graphs = mols_to_nx(smiles_to_mols(test_data))
#     gen_graphs = mols_to_nx(smiles_to_mols([smiles for smiles in smiles_list if smiles is not None]))
#     nspdk_score = compute_nspdk_mmd(test_graphs, gen_graphs,
#                                     metric='nspdk', is_hist=False, n_jobs=20)
#     metrics_dict = moses_statistics
#     metrics_dict['NSPDK'] = nspdk_score
#     wandb.log(metrics_dict)
    
#   def get_fragments(self):
#     print("Generate fragments")
#     train_mol_list = Parallel(n_jobs=8)(delayed(MolFromSmiles)(smiles) for smiles in self.train_smiles_set)
#     if hparams.is_single:
#       fragments = Parallel(n_jobs=8)(delayed(BRICSDecompose)(mol, returnMols=True, singlePass=True) for mol in train_mol_list)
#     else:
#       fragments = Parallel(n_jobs=8)(delayed(BRICSDecompose)(mol, returnMols=True, singlePass=False) for mol in train_mol_list)
#     print("Decomposition done")
#     fragment_list = [frag for frag_list in fragments for frag in list(frag_list)]
#     fragment_smiles_list = Parallel(n_jobs=8)(delayed(MolToSmiles)(frag) for frag in fragment_list)
#     # filter out fragments without attachment points
#     fragment_smiles_list = [frags for frags in fragment_smiles_list if '*' in frags]
#     with open(f'../resource/fragment/{hparams.dataset_name}/fragment_list.txt', 'w') as f:
#       for fragment in fragment_smiles_list:
#         f.write(f'{fragment}\n')
#     print("Fragment generation done!")
  
#   def add_args(parser):
#     parser.add_argument("--string_type", type=str, default="smiles")
#     parser.add_argument("--dataset_name", type=str, default="zinc")
#     parser.add_argument("--fragment_size", type=int, default=1000)
#     parser.add_argument("--random_seed", type=str, default="0")
#     parser.add_argument("--sample_size", type=int, default=10000)
#     parser.add_argument("--random_seed_size", type=int, default=100)
#     parser.add_argument("--get_frag", type=str2bool, default=False)
#     parser.add_argument("--num_workers", type=int, default=6)
#     parser.add_argument("--group", type=str, default='brics')
#     parser.add_argument("--is_single", type=str2bool, default=False)
    

# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
#   BRICSGenerator.add_args(parser)
  
#   hparams = parser.parse_args()
#   model = BRICSGenerator(hparams)
#   if hparams.is_single:
#     file_name = f'../output/brics/{hparams.dataset_name}/{hparams.sample_size}_{hparams.random_seed_size}_smiles_single_list.txt'
#   else:
#     file_name = f'../output/brics/{hparams.dataset_name}/{hparams.sample_size}_{hparams.random_seed_size}_smiles_list.txt'
    
#   if not os.path.isfile(file_name):
#     if hparams.get_frag:
#       model.get_fragments(hparams.dataset_name, hparams.is_single)
    
#     if hparams.is_single:
#       print("Single generation")
#       with open(f'../resource/fragment/{hparams.dataset_name}/fragment_single_list.txt', 'r') as f:
#         fragment_smiles_list = [line.rstrip() for line in f]
#     else:
#       with open(f'../resource/fragment/{hparams.dataset_name}/fragment_list.txt', 'r') as f:
#         fragment_smiles_list = [line.rstrip() for line in f]
      
#     # sample fragments
#     fragment_cands = sample(fragment_smiles_list, hparams.fragment_size)
#     sampled_mols = []
#     for r in tqdm(range(hparams.random_seed_size), desc="Sampling molecules"):
#       random.seed(0x1234+r)
#       fragment_cands_mols = Parallel(n_jobs=8)(delayed(MolFromSmiles)(frag) for frag in fragment_cands)
#       brics_gen = BRICSBuild(fragment_cands_mols)
#       batch_size = int(hparams.sample_size/hparams.random_seed_size)
#       for tm in [next(brics_gen) for x in range(batch_size)]:
#         sampled_mols.append(tm)

#     samples_smiles = Parallel(n_jobs=8)(delayed(MolToSmiles)(mol) for mol in sampled_mols)
#     if hparams.is_single:
#       print("Single file writing")
#     with open(file_name, 'w') as f:
#       for smiles in samples_smiles:
#         f.write(f'{smiles}\n')
  
#   if hparams.is_single:
#     print("Load single file")
#   with open(file_name, 'r') as f:
#     samples_smiles = [line.rstrip() for line in f]
  
#   # check samples (logging)
#   smiles_list = [canonicalize(smiles) for smiles in samples_smiles]
#   model.check_samples(smiles_list)
  