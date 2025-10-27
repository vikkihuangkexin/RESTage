import os
from geneformer import TranscriptomeTokenizer
organ='skin'
tk = TranscriptomeTokenizer({"ID":"ID",
                             "age":"age",
                             "group":"group",
                             "cluster_id_2":"sasp",
                             "ident": "cell_type",
                             'CellID': 'barcode'}, nproc=16)
tk.tokenize_data(f'data_demo/data_impute',
                 f"data_demo/senescence/{organ}/Training",
                 organ,
                 file_format="loom")
####Merge labels file
import pandas as pd
path = f'data_demo/senescence/{organ}/Training'
file_pos = os.path.join(path, 'pos_P.csv')
file_neg = os.path.join(path, 'neg_P.csv')
file_output = os.path.join(path, 'label_ensembl.csv')
try:
    df_pos_raw = pd.read_csv(file_pos, usecols=['ENSEMBL'])
    df_neg_raw = pd.read_csv(file_neg, usecols=['ENSEMBL'])
    df_pos = df_pos_raw.rename(columns={'ENSEMBL': 'SASP_interaction'})
    df_neg = df_neg_raw.rename(columns={'ENSEMBL': 'non_interaction'})
    df_pos = df_pos.reset_index(drop=True)
    df_neg = df_neg.reset_index(drop=True)
    df_combined = pd.concat([df_pos, df_neg], axis=1)
    df_combined.to_csv(file_output, index=False)
