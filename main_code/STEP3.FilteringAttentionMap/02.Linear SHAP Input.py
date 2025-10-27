import numpy as np
import pandas as pd
import os

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
        activate_concat.to_csv(os.path.join(fr"{data_path}/{organ}/shap", 'input',cell_type_A, f'{pivot}_input.csv'))