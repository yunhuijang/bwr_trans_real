# from brics_generator import BreakBRICSBonds, FindBRICSBonds
from rdkit.Chem.BRICS import FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem import MolFromSmiles, MolToSmiles
# VOCAB_SIZE = 100
# PRE_TOKENIZATION = False
# IS_CHARACTER = False
# UniGramTokenizer(VOCAB_SIZE, PRE_TOKENIZATION, IS_CHARACTER, dataset='zinc')
# train_sentence_piece(dataset='qm9', model_type='unigram', vocab_size=40)
# train_sentence_piece(dataset='qm9', model_type='unigram', vocab_size=50)
# train_sentence_piece(dataset='zinc', model_type='bpe', vocab_size=127)
# train_sentence_piece(dataset='zinc', model_type='bpe', vocab_size=200, is_tokenized=True)
# train_sentence_piece(dataset='qm9', model_type='bpe', vocab_size=127, is_tokenized=True)

m = MolFromSmiles('CCCOCCC(=O)c1ccccc1')
m2=BreakBRICSBonds(m, False)
print(MolToSmiles(m2))
