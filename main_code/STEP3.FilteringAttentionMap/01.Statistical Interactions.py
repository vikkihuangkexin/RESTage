import datetime
import cv2
import subprocess
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

organ = 'skin'
output_dir = r'model path/att_map'
full_values = pd.read_csv(os.path.join(output_dir,'cytokine_fullmap.txt'))
full_values.index = full_values['gene_name']
full_values = full_values.iloc[:,2:]
full_values = full_values.loc[(full_values != 0).any(axis=1)]
TCell = full_values.loc[:, [col for col in full_values.columns.to_list() if '-T' in col]]

count_data = map_datas = {'T': TCell}
cy_result = {}
output_dir = os.path.join(output_dir, 'by_map')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for key in list(count_data.keys()):
    counts = count_data[key]
    counts = counts.loc[(counts != 0).any(axis=1)]
    genelist = counts.index
    col = counts.columns.to_list()
    cylist = list(set([bar.split('-')[0] for bar in col]))
    final_mtx = pd.DataFrame(columns=cylist,index=genelist)
    for cy in cylist:
        cy_map = counts.loc[:, [col for col in counts.columns.to_list() if f'{cy}-' in col]]
        cy_map = cy_map.replace(0, float('nan'))
        cy_map = cy_map.applymap(lambda x: np.nan if isinstance(x, str) else x)
        final_mtx[cy] = cy_map.sum(axis=1) / cy_map.count(axis=1)
        final_mtx= final_mtx.replace(float('nan'),0)
    final_mtx.to_csv(os.path.join(output_dir, f'full_attn_{key.replace(r"/", "-")}_mean_test.csv'))
    maps_c = final_mtx

    flat_data = maps_c.unstack()
    sorted_data = flat_data.sort_values(ascending=False)
    # Calculate the top 20% of data
    top_20_percent = int(len(sorted_data) * 0.2)
    top_data = sorted_data[:top_20_percent]
    df_D = pd.DataFrame(top_data)
    df_D.reset_index(inplace=True)
    df_D.columns = ['cy', 'gene', 'Value']
    df_D.to_csv(os.path.join(output_dir, f'att_map_{key}-Cell_Top20InMap.csv'), index=False)
    interaction = df_D.groupby('cy')['gene'].agg(list).to_dict()
    lengths = [len(v) for v in interaction.values()]
    plt.bar(interaction.keys(), lengths)
    plt.title(f'{key}')
    plt.xlabel('cytokine')
    plt.ylabel('gene num')
    plt.xticks([])
    plt.savefig(os.path.join(output_dir, f'{key}-Cell_top20-gene-num.png'))
    plt.close()
    interaction = df_D['gene'].value_counts().sort_values(ascending=False)
    record = {'gene': interaction.index, 'interaction_counts': interaction.values}
    record = pd.DataFrame(record)
    record.to_csv(os.path.join(output_dir, f'{key}_gene_interaction_counts_ByMap.csv'))

    top_20 = interaction[:20]

    plt.figure(figsize=(20, 6))
    plt.bar(top_20.index, top_20.values)
    plt.title(f'{key}')
    plt.xlabel('Gene')
    plt.ylabel('gene num')
    for i in range(min(15, top_20.shape[0])):
        plt.text(i, top_20.values[i], top_20.index[i], ha='center', va='bottom', rotation=45,
                 fontsize=15)
    plt.xticks([])
    plt.savefig(os.path.join(output_dir, f'{key}-Cell_top20-geneNum.png'))
    plt.close()
def _norm(y, filter = False, centralize = False):
    zero_ratio = 0.95
    min_count = 1000
    if filter:
        # filter bad genes
        y = y.loc[(y == 0).mean(axis=1) < zero_ratio]
        # filter bad cell barcodes
        y = y.loc[:, y.sum() >= min_count]
    size_factor = 1E5 / y.sum()
    y *= size_factor
    y = numpy.log2(y + 1)
    if centralize:
        background = y.mean(axis=1)
        y = y.subtract(background, axis=0)
    return y
data_path = r'data_demo/senescence'
organ = 'skin'
data_info = pd.read_csv(f'data_demo/senescence/{organ}/data_information.csv')
cell_types = data_info['cell_type'].dropna().tolist()
threshold = int(data_info['threshold'].tolist()[0])
cell_activate = pd.read_csv(rf'{data_path}/{organ}/no_filtered/max_min_index_{threshold}/geneset_signature.txt',sep='\t',index_col=0)
sasp_activate = pd.read_csv(rf'{data_path}/{organ}/no_filtered/max_min_index_{threshold}/cytosig_activity.txt',sep='\t',index_col=0)
sasp_activate = sasp_activate.T
for cell_type in cell_types:
    cell_type = cell_type.replace('/','-')
    cell_exp = pandas.read_csv(fr"{data_path}/{organ}/T_cell_counts.txt",
                               sep="\t")
    expression = _norm(cell_exp, filter=False, centralize=True)
    expression = expression.T
    activate = pandas.DataFrame(index=expression.index)
    activate.loc[:, f'activate'] = cell_activate['Proliferation']
    pre = []
    cell_type_A = 'Mono' if cell_type.lower().startswith('mono') or cell_type.lower().startswith('macro') else cell_type
    interaction = pd.read_csv(os.path.join(fr"{data_path}/shap",f'att_map_{cell_type_A}-Cell_Top20InMap.csv'))
    cylist = list(set(interaction['cy']))
    if not os.path.exists(os.path.join(fr"{data_path}/shap", 'input',cell_type_A)):
        os.makedirs(os.path.join(fr"{data_path}/shap", 'input',cell_type_A))
    for pivot in cylist:
        activate_C = activate.copy()
        activate_C.loc[:, f'cytosig_{pivot}'] = sasp_activate[pivot]
        Gene_index = list(interaction[interaction['cy'] == pivot]['gene'])
        activate_concat = []
        activate_concat.append(activate_C)
        for gid in Gene_index:
            df = pd.DataFrame({f'gene_{gid}':expression[gid]})
            activate_concat.append(df)
        activate_concat = pd.concat(activate_concat,axis=1)
        activate_concat.to_csv(os.path.join(fr"{data_path}/shap", 'input',cell_type_A, f'{pivot}_input.csv'))
