import os, sys, pandas, numpy, pathlib, getopt
import CytoSig
sys.path.append("data_demo/senescence")
import fun_def
from scipy import stats
from statsmodels.stats.multitest import multipletests

err_tol = 1e-8
verbose_flag = False
inter_threshold = "15"
organ = "skin"
CellType='T'
if not os.path.exists("data_demo/senescence/"+organ+"/no_filtered/max_min_index_"+inter_threshold+"/"):
    os.makedirs("data_demo/senescence/"+organ+"/no_filtered/max_min_index_"+inter_threshold+"/")
fpath = "data_demo/senescence/"+organ+"/no_filtered/max_min_index_"+inter_threshold+"/"
print(fpath)
####更新了4个细胞因子的特征矩阵
#signature = pandas.read_csv("/data_d/ZLX/github_program/CytoSig-master/CytoSig/signature.centroid", sep='\t', index_col=0)
signature = pandas.read_csv("data_demo/senescence/sasp_matrix.txt", sep='\t', index_col=0)
t_cell_exp = pandas.read_csv("data_demo/senescence/"+organ+"/"+CellType+"_cell_counts.txt", sep="\t")
sasp_cell_exp = pandas.read_csv("data_demo/senescence/"+organ+"/SASP_counts.txt", sep="\t")
##去除免疫细胞里的衰老细胞
common_cells = t_cell_exp.columns.intersection(sasp_cell_exp.columns)
t_cell_exp = t_cell_exp.drop(columns=common_cells)
chosen_cell = pandas.read_csv("data_demo/senescence/"+organ+"/cell_interaction/max_min_merge_"+inter_threshold+".txt", sep="\t")
if not "." in t_cell_exp.columns[1]:
    chosen_cell.columns = chosen_cell.columns.str.replace("[.]", "-")
t_cell_exp_2 = t_cell_exp.loc[:, chosen_cell.columns]
expression = fun_def._norm(t_cell_exp_2, filter=False, centralize=True)###对应文件夹filtered和no_filtered
print(expression.shape)
print(chosen_cell.shape)

def profile_geneset_signature(expression,CellType):
    signature = []
    fin = open(f'data_demo/Tres_{CellType}.kegg')
    for l in fin:
        fields = l.strip().split('\t')

        s = fields[2:]
        signature.append(pandas.Series(numpy.ones(len(s)), index=s, name=fields[0]))
    fin.close()

    signature = pandas.concat(signature, axis=1, join='outer', sort=False)
    signature.fillna(0, inplace=True)

    common = expression.index.intersection(signature.index)
    signature, expression = signature.loc[common], expression.loc[common]

    background = signature.mean(axis=1)
    background.name = 'study bias'

    X = signature.loc[:, [f'_{CellType}_CELL_ACTIVATION']].mean(axis=1)
    X.name = 'Proliferation'

    X = pandas.concat([background, X], axis=1, join='inner')

    result = CytoSig.ridge_significance_test(X, expression, alpha=0, alternative="two-sided", nrand=0,
                                             verbose=verbose_flag)

    return result[2].loc['Proliferation']


def interaction_test(expression, X, y):
    signal = X.loc[:, 'pivot']
    failed = []
    merge = []
    for gid, arr in expression.iterrows():
        X.loc[:, 'partner'] = arr
        X.loc[:, 'interaction'] = arr * signal
        # other covariates have no sufficient variation
        if arr.std() < err_tol or X.loc[:, 'interaction'].std() < err_tol: continue
        try:
            y = pandas.DataFrame(y)
            result = CytoSig.ridge_significance_test(X, y, alpha=0, alternative="two-sided", nrand=0,
                                                     flag_normalize=False, verbose=verbose_flag)
        except ArithmeticError:
            failed.append(gid)
            continue
        tvalue = result[2].loc['interaction'].iloc[0]
        pvalue = result[3].loc['interaction'].iloc[0]

        merge.append(pandas.Series([tvalue, pvalue], index=['t', 'p'], name=gid))

    result = pandas.concat(merge, axis=1, join='inner').transpose()
    result['q'] = multipletests(result['p'], method='fdr_bh')[1]

    return result

###############################################################
# compute signaling activity
result_signaling = fun_def.ridge_significance_test(signature, expression, alpha=1E4, verbose=verbose_flag)
# get the z-scores
result_signaling = result_signaling[2]
result_signaling.to_csv(fpath+"cytosig_activity.txt", sep="\t")
###############################################################
# compute proliferation signature
result_prolifertion = profile_geneset_signature(expression,CellType)
result_prolifertion.to_csv(fpath+"geneset_signature.txt", sep="\t")