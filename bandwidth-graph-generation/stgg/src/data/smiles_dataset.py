from rdkit import Chem
import enum
from deepsmiles import Converter
import selfies as sf
from joblib import Parallel, delayed
import sentencepiece as spm
from collections import defaultdict
import pandas as pd
import json
from math import log
import copy
import os
from tqdm import tqdm

PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"
TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    "#",
    "(",
    ")",
    "-",
    "/",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "=",
    "Br",
    "C",
    "Cl",
    "F",
    "I",
    "N",
    "O",
    "P",
    "S",
    "[C@@H]",
    "[C@@]",
    "[C@H]",
    "[C@]",
    "[CH-]",
    "[CH2-]",
    "[H]",
    "[N+]",
    "[N-]",
    "[NH+]",
    "[NH-]",
    "[NH2+]",
    "[NH3+]",
    "[O+]",
    "[O-]",
    "[OH+]",
    "[P+]",
    "[P@@H]",
    "[P@@]",
    "[P@]",
    "[PH+]",
    "[PH2]",
    "[PH]",
    "[S+]",
    "[S-]",
    "[S@@+]",
    "[S@@]",
    "[S@]",
    "[SH+]",
    "[n+]",
    "[n-]",
    "[nH+]",
    "[nH]",
    "[o+]",
    "[s+]",
    "\\",
    "c",
    "n",
    "o",
    "s",
]
TOKENS_SELFIES = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
# in QM9
'[#Branch1]', '[#Branch2]', '[#C]', '[#N]', 
'[/C]', '[/Cl]', '[/NH1+1]', '[/N]', '[/O-1]', '[/O]', '[/S-1]', 
'[/S]', '[=Branch1]', '[=Branch2]', '[=C]', '[=N+1]', '[=NH1+1]', 
'[=NH2+1]', '[=N]', '[=O]', '[=Ring1]', '[=Ring2]', '[=S@@]', 
'[=S]', '[Br]', '[Branch1]', '[Branch2]', '[C@@H1]', '[C@@]', 
'[C@H1]', '[C@]', '[C]', '[Cl]', '[F]', '[I]', '[N+1]', '[N-1]', 
'[NH1+1]', '[NH1-1]', '[NH1]', '[NH2+1]', '[NH3+1]', '[N]', '[O-1]', 
'[O]', '[P@@]', '[P]', '[Ring1]', '[Ring2]', '[S-1]', '[S@@]', '[S@]', 
'[S]', '[\\C]', '[\\N]', '[\\O-1]', '[\\O]', '[\\S]',
# in ZINC
'[PH1+1]', '[S@@+1]', '[=S@]', '[\\S@]', '[S+1]', '[=SH1+1]', '[\\N+1]', '[/O+1]', 
'[/C@@]', '[-\\Ring1]', '[/NH1-1]', '[\\NH1]', '[-/Ring2]', '[/NH1]', '[\\NH2+1]', 
'[/C@]', '[=O+1]', '[=P]', '[PH1]', '[\\S-1]', '[\\I]', '[/C@H1]', '[=PH2]', 
'[/C@@H1]', '[/F]', '[/N+1]', '[\\Br]', '[\\N-1]', '[\\Cl]', '[\\NH1+1]', '[\\F]', 
'[=P@@]', '[P@@H1]', '[P+1]', '[=S+1]', '[\\C@H1]', '[=N-1]', '[CH2-1]', '[-/Ring1]', 
'[CH1-1]', '[/NH2+1]', '[\\C@@H1]', '[/S@]', '[=OH1+1]', '[=P@]', '[/Br]', '[/N-1]', 
'[P@]', '[#N+1]']
TOKENS_DEEPSMILES = TOKENS + ['%20', '%22', '%28', '%12', 
'%15', '%13', '%10', '%16', '%11', 
'%14', '%21', '%18', '%17', '%19']
MAX_LEN = 250

# set hyperparameters for BPE / UNI
VOCAB_SIZE = 100
PRE_TOKENIZATION = False
IS_CHARACTER = True
pre_tokenization_dict = {True: 'pre', False: 'wopre'}
is_character_dict = {True: 'char', False:'token'}
vocab_size_dict = {'qm9': 127, 'zinc': 200}
# SPM tokens
# SPM_TOKENS_DICT = defaultdict()
# for dataset in ['qm9', 'zinc']:
#     for model_type in ['unigram', 'bpe']:
#         key = f'{dataset}_{model_type}'
#         SPM_TOKENS_DICT[key] = {}
#         sp = spm.SentencePieceProcessor(model_file=f"../resource/spm_tokenizer/{dataset}/{dataset}_{model_type}_{vocab_size_dict[dataset]}.model")
#         SPM_TOKENS_DICT[key]['sp'] = sp
#         tokens_spm = [sp.IdToPiece(ids) for ids in range(sp.GetPieceSize())]
#         tokens_spm.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN])
#         SPM_TOKENS_DICT[key]['tokens'] = tokens_spm

# SPM_BPE tokens & merges
# def load_bpe(vocab_size, pre_tokenization, is_character):
#     pre = pre_tokenization_dict[pre_tokenization]
#     is_char = is_character_dict[is_character]
#     with open(f"../resource/tokenizer/qm9/merges_{vocab_size}_{pre}_{is_char}.txt", 'r') as merges_file:
#         merges = merges_file.read()
#         merges_dict = eval(merges)
#         MERGES_QM9 = {eval(key): value for key, value in merges_dict.items()}

#     with open(f"../resource/tokenizer/zinc/merges_{vocab_size}_{pre}_{is_char}.txt", 'r') as merges_file:
#         merges = merges_file.read()
#         merges_dict = eval(merges)
#         MERGES_ZINC = {eval(key): value for key, value in merges_dict.items()}

#     with open(f"../resource/tokenizer/qm9/tokens_bpe_{vocab_size}_{pre}_{is_char}.txt", 'r') as tokens_file:
#         tokens = tokens_file.readlines()
#         TOKENS_BPE_QM9 = [token.rstrip('\n') for token in tokens]
#         TOKENS_BPE_QM9.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN])

#     with open(f"../resource/tokenizer/zinc/tokens_bpe_{vocab_size}_{pre}_{is_char}.txt", 'r') as tokens_file:
#         tokens = tokens_file.readlines()
#         TOKENS_BPE_ZINC = [token.rstrip('\n') for token in tokens]
#         TOKENS_BPE_ZINC.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN])

#     return MERGES_QM9, MERGES_ZINC, TOKENS_BPE_QM9, TOKENS_BPE_ZINC

# # SPM_UNI tokens & models
# def load_unigram(vocab_size, pre_tokenization, is_character):
#     pre = pre_tokenization_dict[pre_tokenization]
#     is_char = is_character_dict[is_character]
#     with open(f"../resource/tokenizer/qm9/model_{vocab_size}_{pre}_{is_char}.txt", 'r') as model_file:
#         model = model_file.read()
#         model_dict = eval(model)
#         MODEL_QM9 = model_dict

#     with open(f"../resource/tokenizer/qm9/tokens_uni_{vocab_size}_{pre}_{is_char}.txt", 'r') as tokens_file:
#         tokens = tokens_file.readlines()
#         TOKENS_UNI_QM9 = [token.rstrip('\n') for token in tokens]
#         TOKENS_UNI_QM9.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN])

#     with open(f"../resource/tokenizer/zinc/model_{vocab_size}_{pre}_{is_char}.txt", 'r') as model_file:
#         model = model_file.read()
#         model_dict = eval(model)
#         MODEL_ZINC = model_dict

#     with open(f"../resource/tokenizer/zinc/tokens_uni_{vocab_size}_{pre}_{is_char}.txt", 'r') as tokens_file:
#         tokens = tokens_file.readlines()
#         TOKENS_UNI_ZINC = [token.rstrip('\n') for token in tokens]
#         TOKENS_UNI_ZINC.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN])

#     return MODEL_QM9, MODEL_ZINC, TOKENS_UNI_QM9, TOKENS_UNI_ZINC


# MERGES_QM9, MERGES_ZINC, TOKENS_BPE_QM9, TOKENS_BPE_ZINC = load_bpe(VOCAB_SIZE, PRE_TOKENIZATION, IS_CHARACTER)
# BPE_TOKENS_DICT = {'qm9': {'merges': MERGES_QM9, 'tokens': TOKENS_BPE_QM9}, 'zinc': {'merges': MERGES_ZINC, 'tokens': TOKENS_BPE_ZINC}}
# MODEL_QM9, MODEL_ZINC, TOKENS_UNI_QM9, TOKENS_UNI_ZINC = load_unigram(VOCAB_SIZE, PRE_TOKENIZATION, IS_CHARACTER)
# UNI_TOKENS_DICT = {'qm9': {'model': MODEL_QM9, 'tokens': TOKENS_UNI_QM9}, 'zinc': {'model': MODEL_ZINC, 'tokens': TOKENS_UNI_ZINC}}


@enum.unique
class TokenType(enum.IntEnum):
    ATOM = 1
    BOND = 2
    BRANCH_START = 3
    BRANCH_END = 4
    RING_NUM = 5
    SPECIAL = 6


ORGANIC_ATOMS = "B C N O P S F Cl Br I * b c n o s p".split()

def token_to_id(tokens):
    return {token: tokens.index(token) for token in tokens}

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def split_for_nlp_tokenizer(smiles):
    smiles = iter(smiles)
    tokens = []
    peek = None
    while True:
        char = peek if peek else next(smiles, "")
        peek = None
        if not char:
            break

        if char == "[":
            token = char
            for char in smiles:
                token += char
                if char == "]":
                    break

        elif char in ORGANIC_ATOMS:
            peek = next(smiles, "")
            if char + peek in ORGANIC_ATOMS:
                token = char + peek
                peek = None
            else:
                token = char

        elif char == "%":
            token = char + next(smiles, "") + next(smiles, "")

        elif char in "-=#$:.()%/\\" or char.isdigit():
            token = char
        else:
            raise ValueError(f"Undefined tokenization for chararacter {char}")

        tokens.append(token)
    return tokens

class UniGramTokenizer():
    def __init__(self, vocab_size=VOCAB_SIZE, pre_tokenization=PRE_TOKENIZATION, is_character=IS_CHARACTER, dataset='qm9'):
        self.vocab_size = vocab_size
        self.pre_tokenization = pre_tokenization
        self.dataset = dataset
        self.is_character = is_character
        self.model = UNI_TOKENS_DICT[self.dataset]['model']
        tokens = UNI_TOKENS_DICT[self.dataset]['tokens']
        tokens.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN])
        self.tokens = tokens
        key = f'{self.dataset}_unigram'
        self.sp = SPM_TOKENS_DICT[key]['sp']

        # if not os.path.isfile(f"../resource/tokenizer/{self.dataset}/model_{self.vocab_size}_{pre_tokenization_dict[self.pre_tokenization]}_{is_character_dict[self.is_character]}.txt"):
        #     self.train_tokenizer_with_unigram()

    def encode_word(self, word, model):
        best_segmentations = [{"start": 0, "score": 1}] + [
            {"start": None, "score": None} for _ in range(len(word))
        ]
        for start_idx in range(len(word)):
            # This should be properly filled by the previous steps of the loop
            best_score_at_start = best_segmentations[start_idx]["score"]
            for end_idx in range(start_idx + 1, len(word) + 1):
                token = word[start_idx:end_idx]
                if token in model and best_score_at_start is not None:
                    score = model[token] + best_score_at_start
                    # If we have found a better segmentation ending at end_idx, we update
                    if (
                        best_segmentations[end_idx]["score"] is None
                        or best_segmentations[end_idx]["score"] > score
                    ):
                        best_segmentations[end_idx] = {"start": start_idx, "score": score}

        segmentation = best_segmentations[-1]
        if segmentation["score"] is None:
            # We did not find a tokenization of the word -> unknown
            return ["<unk>"], 0

        score = segmentation["score"]
        start = segmentation["start"]
        end = len(word)
        tokens = []
        while start != 0:
            tokens.insert(0, word[start:end])
            next_start = best_segmentations[start]["start"]
            end = start
            start = next_start
        tokens.insert(0, word[start:end])
        return tokens, score

    def train_tokenizer_with_unigram(self):
        data = pd.read_csv(f'../resource/data/{self.dataset}/train_val.txt')
        data.columns =['smiles']
        if self.pre_tokenization:
            data['tokens'] = data['smiles'].apply(lambda x: self.sp.encode_as_pieces(x))
            data['ids'] = data['smiles'].apply(lambda x: self.sp.encode_as_ids(x))
            word_freqs = dict(data['tokens'].explode().value_counts())
        else:
            word_freqs = {key: 1 for key in data['smiles']}

        char_freqs = defaultdict(int)
        subwords_freqs = defaultdict(int)
        if self.is_character:
            for word, freq in tqdm(word_freqs.items(), desc="subword generation by character"):
                for i in range(len(word)):
                    char_freqs[word[i]] += freq
                    # Loop through the subwords of length at least 2
                    for j in range(i + 2, len(word) + 1):
                        subwords_freqs[word[i:j]] += freq
        else:
            for word, freq in tqdm(word_freqs.items(), desc="subword generation by token"):
                tokens = split_for_nlp_tokenizer(word)
                for i, token in enumerate(tokens):
                    char_freqs[token] += freq
                    for j in range(i+2, len(tokens)+1):
                        subwords_freqs[''.join(tokens[i:j])] += freq

        sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)

        token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
        token_freqs = {token: freq for token, freq in token_freqs}

        total_sum = sum([freq for token, freq in token_freqs.items()])
        model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

        percent_to_remove = 0.1
        while len(model) > self.vocab_size:
            scores = self.compute_scores(model, word_freqs)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            # Remove percent_to_remove tokens with the lowest scores.
            for i in range(int(len(model) * percent_to_remove)):
                _ = token_freqs.pop(sorted_scores[i][0])

            total_sum = sum([freq for token, freq in token_freqs.items()])
            model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

        tokens = list(model.keys())

        with open(f"../resource/tokenizer/{self.dataset}/model_{self.vocab_size}_{pre_tokenization_dict[self.pre_tokenization]}_{is_character_dict[self.is_character]}.txt", 'w') as model_file:
            model_file.write(json.dumps(model))
        with open(f"../resource/tokenizer/{self.dataset}/tokens_uni_{self.vocab_size}_{pre_tokenization_dict[self.pre_tokenization]}_{is_character_dict[self.is_character]}.txt", 'w') as vocab_file:
            for line in tokens:
                vocab_file.write(f"{line}\n")

        return model, tokens

    def compute_loss(self, model, word_freqs):
        loss = 0
        for word, freq in word_freqs.items():
            _, word_loss = self.encode_word(word, model)
            loss += freq * word_loss
        return loss

    def compute_scores(self, model, word_freqs):
        scores = {}
        model_loss = self.compute_loss(model, word_freqs)
        for token, score in model.items():
            # We always keep tokens of length 1
            if len(token) == 1:
                continue
            model_without_token = copy.deepcopy(model)
            _ = model_without_token.pop(token)
            scores[token] = self.compute_loss(model_without_token, word_freqs) - model_loss
        return scores

    def tokenize_uni(self, smiles):

        TOKEN2ID = token_to_id(self.tokens)
        if self.pre_tokenization:
            pre_tokenized_text = self.sp.encode_as_pieces(smiles)
            encoded_words = [self.encode_word(word, self.model)[0] for word in pre_tokenized_text]
        else:
            if self.is_character:
                encoded_words = [self.encode_word(word, self.model)[0] for word in smiles]
            else:
                encoded_words = [self.encode_word(word, self.model)[0] for word in split_for_nlp_tokenizer(smiles)]

        tokens = [TOKEN2ID[BOS_TOKEN]]
        token_strs = sum(encoded_words, [])
        tokens.extend([TOKEN2ID[token] for token in token_strs])
        tokens.append(TOKEN2ID[EOS_TOKEN])
        return tokens

class BPETokenizer():
    def __init__(self, vocab_size=VOCAB_SIZE, pre_tokenization=PRE_TOKENIZATION, is_character=False, dataset='qm9'):
        self.vocab_size = vocab_size
        self.pre_tokenization = pre_tokenization
        self.dataset = dataset
        self.is_character = is_character
        self.merges = BPE_TOKENS_DICT[self.dataset]['merges']
        self.tokens = BPE_TOKENS_DICT[self.dataset]['tokens']
        self.tokens.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN])
        key = f'{self.dataset}_unigram'
        self.sp = SPM_TOKENS_DICT[key]['sp']
        # if not os.path.isfile(f"../resource/tokenizer/{self.dataset}/merges_{self.vocab_size}_{pre_tokenization_dict[self.pre_tokenization]}_{is_character_dict[self.is_character]}.txt"):
        #     self.train_tokenizer_with_bpe()

    def compute_pair_freqs(self, splits, word_freqs):
        # used in train_tokenizer_with_bpe
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b, splits, word_freqs):
        # used in train_tokenizer_with_bpe
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def train_tokenizer_with_bpe(self):
        '''
        is_character: to split character by character or by token (e.g., [NH+])
        '''
        # generate merges and tokens file for BPE
        data = pd.read_csv(f'../resource/data/{self.dataset}/train_val.txt')
        data.columns =['smiles']
        # pre-tokenization
        if self.pre_tokenization:
            data['ids'] = data['smiles'].apply(lambda x: self.sp.encode_as_ids(x))
            word_freqs = dict(data['tokens'].explode().value_counts())
        else:
            word_freqs = {key: 1 for key in data['smiles']}

        key_list = [list(key) for key in word_freqs.keys()]
        vocab = list({char for char_list in key_list for char in char_list})
        splits = {word: [c for c in word] for word in word_freqs.keys()}
        if not self.is_character:
            vocab = TOKENS
            splits = {word: split_for_nlp_tokenizer(word) for word in word_freqs.keys()}


        pair_freqs = self.compute_pair_freqs(splits, word_freqs)
        merges = defaultdict()
        while len(vocab) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs(splits, word_freqs)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            if best_pair == '':
                print(f"Too large vocab size: break at {len(vocab)} before vocab size {self.vocab_size}")
                break
            splits = self.merge_pair(*best_pair, splits, word_freqs)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
        # save generataed merges and tokens
        merges_str = {str(key):value for key, value in merges.items()}
        with open(f"../resource/tokenizer/{self.dataset}/merges_{self.vocab_size}_{pre_tokenization_dict[self.pre_tokenization]}_{is_character_dict[self.is_character]}.txt", 'w') as merges_file:
            merges_file.write(json.dumps(merges_str))
        with open(f"../resource/tokenizer/{self.dataset}/tokens_bpe_{self.vocab_size}_{pre_tokenization_dict[self.pre_tokenization]}_{is_character_dict[self.is_character]}.txt", 'w') as vocab_file:
            for line in vocab:
                vocab_file.write(f"{line}\n")
        return merges, vocab

    def tokenize_bpe(self, smiles):
        TOKEN2ID = token_to_id(self.tokens)
        if self.pre_tokenization:
            pre_tokenized_text = self.sp.encode_as_pieces(smiles)
            splits = [[l for l in word] for word in pre_tokenized_text]
        else:
            if self.is_character:
                splits = [[l for l in word] for word in smiles]
            else:
                splits = [split_for_nlp_tokenizer(smiles)]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split
        tokens = [TOKEN2ID[BOS_TOKEN]]
        token_strs = sum(splits, [])
        tokens.extend([TOKEN2ID[token] for token in token_strs])
        tokens.append(TOKEN2ID[EOS_TOKEN])
        return tokens

def tokenize_spm(smiles, dataset='qm9', model_type='bpe'):
    key = f'{dataset}_{model_type}'
    sp = SPM_TOKENS_DICT[key]['sp']
    tokens_spm = SPM_TOKENS_DICT[key]['tokens']
    TOKEN2ID = token_to_id(tokens_spm)
    tokens = [TOKEN2ID[BOS_TOKEN]]
    tokens.extend(sp.encode_as_ids(smiles))
    tokens.append(TOKEN2ID[EOS_TOKEN])
    return tokens

# def tokenize_brics(smiles):
#     with open(f'../resource/fragment/{dataset}/fragment_single_list.txt', 'r') as f:
#     TOKENS_BRICS = [line.rstrip() for line in f]

def tokenize_selfies(selfies):
    selfies_split = selfies.split("]")[:-1]
    tokens = ["[bos]"]
    tokens.extend([selfies + "]" for selfies in selfies_split])
    tokens.append("[eos]")
    TOKEN2ID = token_to_id(TOKENS_SELFIES)
    return [TOKEN2ID[token] for token in tokens]

def tokenize_deepsmiles(deep_smiles):
    deep_smiles = iter(deep_smiles)
    tokens = ["[bos]"]
    peek = None
    while True:
        char = peek if peek else next(deep_smiles, "")
        peek = None
        if not char:
            break

        if char == "[":
            token = char
            for char in deep_smiles:
                token += char
                if char == "]":
                    break

        elif char in ORGANIC_ATOMS:
            peek = next(deep_smiles, "")
            if char + peek in ORGANIC_ATOMS:
                token = char + peek
                peek = None
            else:
                token = char

        elif char == "%":
            peek = next(deep_smiles, "")
            if peek == '(':
                token = char + peek + next(deep_smiles, "") + next(deep_smiles, "") + next(deep_smiles, "") + next(deep_smiles, "")
            else:
                token = char + peek + next(deep_smiles, "")

        elif char in "-=#$:.)%/\\" or char.isdigit():
            token = char
        else:
            raise ValueError(f"Undefined tokenization for chararacter {char}")

        tokens.append(token)

    tokens.append("[eos]")
    TOKEN2ID = token_to_id(TOKENS_DEEPSMILES)
    return [TOKEN2ID[token] for token in tokens]

def tokenize(smiles):
    smiles = iter(smiles)
    tokens = ["[bos]"]
    peek = None
    while True:
        char = peek if peek else next(smiles, "")
        peek = None
        if not char:
            break

        if char == "[":
            token = char
            for char in smiles:
                token += char
                if char == "]":
                    break

        elif char in ORGANIC_ATOMS:
            peek = next(smiles, "")
            if char + peek in ORGANIC_ATOMS:
                token = char + peek
                peek = None
            else:
                token = char

        elif char == "%":
            token = char + next(smiles, "") + next(smiles, "")
            print(token)
            print(TOKEN2ID[token])
            assert False

        elif char in "-=#$:.()%/\\" or char.isdigit():
            token = char
        else:
            raise ValueError(f"Undefined tokenization for chararacter {char}")

        tokens.append(token)

    tokens.append("[eos]")
    TOKEN2ID = token_to_id(TOKENS)
    return [TOKEN2ID[token] for token in tokens]

def untokenize(sequence, string_type, dataset):
    if string_type in ['spm_unigram', 'spm_bpe']:
        model_type = string_type.split('_')[1]
        tokens = SPM_TOKENS_DICT[f'{dataset}_{model_type}']['tokens']
        ID2TOKEN = id_to_token(tokens)
    elif string_type == 'selfies':
        ID2TOKEN = id_to_token(TOKENS_SELFIES)
    elif string_type == 'deep_smiles':
        ID2TOKEN = id_to_token(TOKENS_DEEPSMILES)
    elif string_type == 'smiles':
        ID2TOKEN = id_to_token(TOKENS)
    elif string_type == 'bpe':
        tokens = BPE_TOKENS_DICT[dataset]['tokens']
        ID2TOKEN = id_to_token(tokens)
    elif string_type == 'uni':
        tokens = UNI_TOKENS_DICT[dataset]['tokens']
        ID2TOKEN = id_to_token(tokens)
    else:
        raise ValueError(f"Undefined string type {string_type}")


    tokens = [ID2TOKEN[id_] for id_ in sequence]
    if tokens[0] != "[bos]":
        return ""
    elif "[eos]" not in tokens:
        return ""

    tokens = tokens[1 : tokens.index("[eos]")]
    return "".join(tokens)


DATA_DIR = "bandwidth-graph-generation/stgg/resource/data"

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc"
    simple = True

    def __init__(self, split, string_type='smiles'):
        self.string_type = string_type

        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        if string_type in ['smiles', 'spm_unigram', 'spm_bpe', 'bpe', 'bpe_zinc', 'uni', 'uni_zinc']:
            string_list = smiles_list
        elif string_type == 'selfies':
            string_list = Parallel(n_jobs=8)(delayed(sf.encoder)(smiles) for smiles in smiles_list)
        elif string_type == 'deep_smiles':
            converter = Converter(rings=True, branches=True)
            string_list = Parallel(n_jobs=8)(delayed(converter.encode)(smiles) for smiles in smiles_list)
        else :
            raise ValueError(f"Undefined string type {string_type}")
        self.smiles_list = string_list
        self.bpe_tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, pre_tokenization=PRE_TOKENIZATION, is_character=IS_CHARACTER, dataset=self.raw_dir.split('/')[-1])
        self.uni_tokenizer = UniGramTokenizer(vocab_size=VOCAB_SIZE, pre_tokenization=PRE_TOKENIZATION, is_character=IS_CHARACTER, dataset=self.raw_dir.split('/')[-1])

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        if self.string_type == 'smiles':
            mol = Chem.MolFromSmiles(smiles)
            
            if self.simple:
                Chem.Kekulize(mol)

            smiles = Chem.MolToSmiles(mol)
            return torch.LongTensor(tokenize(smiles))

        elif self.string_type == 'selfies':
            return torch.LongTensor(tokenize_selfies(smiles))
        elif self.string_type == 'deep_smiles':
            return torch.LongTensor(tokenize_deepsmiles(smiles))
        elif self.string_type == 'spm_unigram':
            return torch.LongTensor(tokenize_spm(smiles, self.raw_dir.split('/')[-1], model_type='unigram'))
        elif self.string_type == 'spm_bpe':
            return torch.LongTensor(tokenize_spm(smiles, self.raw_dir.split('/')[-1], model_type='bpe'))
        elif self.string_type == 'bpe':
            return torch.LongTensor(self.bpe_tokenizer.tokenize_bpe(smiles))
        elif self.string_type == 'uni':
            return torch.LongTensor(self.uni_tokenizer.tokenize_uni(smiles))
        else:
            raise ValueError(f"Undefined string type {self.string_type}")
    
    def get_tokens(self):
        if self.string_type == 'selfies':
            symbols = sf.get_alphabet_from_selfies(self.smiles_list)
            return symbols

        
class QM9Dataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/qm9"
    simple = True


class SimpleMosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"
    simple = True


class MosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"
    simple = False