import os
import sys
import pandas as pd
import numpy as np
import pathlib
import getopt
import string
import scipy

sys.path.append("data_demo/senescence")
import fun_def
import anndata

organ = "skin"

def new_split_subsets(y):
    # split column names by underscore and group columns by the second field (sample id)
    flag_group = [v.split('_')[1] for v in y.columns]
    result = y.groupby(flag_group, axis=1)
    return result

# create output folder for per-sample cell interaction results
out_dir = f"data_demo/senescence/{organ}/cell_interaction/sample"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fpath = out_dir + "/"
verbose_flag = False

# read expression matrices for T cells and SASP cells
t_cell_exp = pd.read_csv(f"data_demo/senescence/{organ}/T_cell_counts.txt", sep="\t", index_col=0)
sasp_cell_exp = pd.read_csv(f"data_demo/senescence/{organ}/SASP_counts.txt", sep="\t", index_col=0)

# remove overlapping cells (cells labeled both as T and SASP)
common_cells = t_cell_exp.columns.intersection(sasp_cell_exp.columns)
if len(common_cells) > 0:
    t_cell_exp = t_cell_exp.drop(columns=common_cells)

# normalize using project function (assumes fun_def.my_norm returns a DataFrame)
t_cell_exp = fun_def.my_norm(t_cell_exp, filter=False)
sasp_cell_exp = fun_def.my_norm(sasp_cell_exp, filter=False)

# split matrices by sample (group columns by sample id extracted from column names)
t_cell_exp_group = new_split_subsets(t_cell_exp)
sasp_cell_exp_group = new_split_subsets(sasp_cell_exp)

# keep only groups present in both datasets
common_group = set(t_cell_exp_group.groups.keys()).intersection(set(sasp_cell_exp_group.groups.keys()))

# read ligand-receptor pairs (expects file with at least two columns for ligand and receptor)
ligand_and_receptor = pd.read_csv("data_demo/human_lr_pair.txt", sep="\t", header=None)
ligand_and_receptor_list = ligand_and_receptor.iloc[:, 1:3].copy()
ligand_and_receptor_list.columns = ['ligand', 'receptor']

# compute per-sample cell communication scores
for name in sorted(common_group):
    t_sub = t_cell_exp.loc[:, t_cell_exp_group.get_group(name).columns]
    sasp_sub = sasp_cell_exp.loc[:, sasp_cell_exp_group.get_group(name).columns]

    # convert to AnnData objects with cells as observations
    adata_t = anndata.AnnData(t_sub.T)
    adata_sasp = anndata.AnnData(sasp_sub.T)

    # initialize aggregated score matrix (SASP cells x T cells)
    merge = np.zeros((sasp_sub.shape[1], t_sub.shape[1]), dtype=float)

    # iterate receptor-ligand pairs and sum pairwise scores
    for i in range(ligand_and_receptor_list.shape[0]):
        cytokine = ligand_and_receptor_list.iloc[i:(i + 1), :]

        # check overlap of ligand in sender (sasp_sub) and receptor in receiver (t_sub)
        ligand_gene = cytokine.iloc[0, 0]
        receptor_gene = cytokine.iloc[0, 1]
        ligand_in_sasp = ligand_gene in sasp_sub.index
        receptor_in_t = receptor_gene in t_sub.index

        if ligand_in_sasp and receptor_in_t:
            lr_list = cytokine
            S = fun_def.cell_communication(adata_t, adata_sasp, lr_list)
            merge = merge + S

    # save aggregated matrix with proper row/column labels
    data = np.array(merge)
    cor_result = pd.DataFrame(data=data, columns=t_sub.columns)
    cor_result.index = sasp_sub.columns
    cor_result.to_csv(f"{fpath}{name}.txt", sep="\t")
