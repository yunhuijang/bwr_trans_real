import pandas as pd
import json
from GDSS_new.utils.mol_utils import load_smiles

dataset = 'zinc'
col = 'smiles'
train_val_df = pd.read_csv(f'stgg/resource/data/{dataset}/train_val.txt', header=None)
train_val_df.columns = ['smiles']
test_df = pd.read_csv(f'stgg/resource/data/{dataset}/test.txt', header=None)
test_df.columns = ['smiles']
total_df = pd.read_csv(f'stgg/resource/data/{dataset}/all.txt', header=None)
total_df.columns = ['smiles']

# gdss_df의 형태로 total_df의 데이터를 합치기

gdss_df = pd.read_csv(f'GDSS_new/data/{dataset}250k.csv')
gdss_df_filtered = gdss_df.loc[gdss_df[col].isin(total_df.smiles)]
gdss_df_filtered = gdss_df_filtered.drop(['Unnamed: 0'], axis=1)
gdss_df_filtered = gdss_df_filtered.reset_index(drop=True)
gdss_df_filtered.to_csv(f'GDSS_new/data/{dataset}250k_new.csv')
# val_idx 뽑기
val_idx_df = gdss_df_filtered.loc[gdss_df_filtered[col].isin(test_df.smiles)]
idx_dict = {"valid_idxs": list(val_idx_df.index)}
with open(f'GDSS_new/data/valid_idx_{dataset}250k_new.json','w') as f:
    json.dump(idx_dict,f)
