import pandas, numpy, sys, os
import ridge_significance

from glob import glob
from scipy import io
fpath = "/data03/ZLX/senescence_test/liver/"
import time
import resource
import os, sys, pandas, numpy, pathlib, getopt
import CytoSig


from scipy import stats
from statsmodels.stats.multitest import multipletests

err_tol = 1e-8
verbose_flag = False


def dataframe_to_array(x, dtype=None):
    """ convert data frame to numpy matrix in C order, in case numpy and gsl use different matrix order """

    x = x.to_numpy(dtype=dtype)
    if x.flags.f_contiguous: x = numpy.array(x, order='C')
    return x


def array_to_dataframe(x, row_names, col_names):
    if type(x) == list:
        for i in range(len(x)): x[i] = pandas.DataFrame(x[i], index=row_names, columns=col_names)
    else:
        x = pandas.DataFrame(x, index=row_names, columns=col_names)

    return x

def ridge_significance_test(X, Y, alpha, alternative="two-sided", nrand=1000, cnt_thres=10, flag_normalize=True,
                            flag_const=False, verbose=True):
    # convert X from series to data frame if necessary
    if type(X) == pandas.Series: X = pandas.DataFrame(X)

    # if X Y index doesn't align
    if X.index.shape[0] != Y.index.shape[0] or sum(X.index != Y.index) > 0:
        common = Y.index.intersection(X.index)

        if common.shape[0] < cnt_thres:
            sys.stderr.write('X dimension %d < %s\n' % (common.shape[0], cnt_thres))
            sys.exit(1)

        Y, X = Y.loc[common], X.loc[common]

    if flag_normalize:
        # normalize to zero mean and unit variation
        X = (X - X.mean()) / X.std()
        X = X.fillna(0) ##################
        X = X.loc[:, ~(X == 0).all()] ##############除去全为0的列
        Y = (Y - Y.mean()) / Y.std()

    if flag_const: X['const'] = 1

    # decorate the results later
    X_columns = X.columns
    Y_columns = Y.columns

    # turn pandas data frame in column major order to numpy 2d array in row major order (compatible with GSL)
    X = dataframe_to_array(X)
    Y = dataframe_to_array(Y)

    # start computation by different alpha formats (arrays or single values)
    if type(alpha) in [list, numpy.ndarray]:
        # if a series of alpha values are considered
        N_alpha = len(alpha)

        result = [None] * N_alpha

        for i in range(N_alpha):
            result[i] = ridge_significance.fit(X, Y, alpha[i], alternative, nrand, verbose)
            result[i] = array_to_dataframe(result[i], X_columns, Y_columns)
    else:
        # if only one alpha value
        result = ridge_significance.fit(X, Y, alpha, alternative, nrand, verbose)
        result = array_to_dataframe(result, X_columns, Y_columns)

    return result

def cell_communication(adata_t,adata_sasp, lr_list):
    ligand_gene = lr_list.loc[:, 'ligand'].drop_duplicates().tolist()
    receptor_gene = lr_list.loc[:, 'receptor'].drop_duplicates().tolist()
    ligand_expression = adata_sasp[:, ligand_gene].X
    receptor_expression = adata_t[:, receptor_gene].X
    betas = numpy.ndarray((len(adata_t), len(receptor_gene)))
    for i in range(betas.shape[1]):
        betas[:, i] = phi_exp(receptor_expression[:, i], 1.0, 1.0, 1)
    beta = numpy.mean(betas, axis=1)
    alpha = phi_exp(ligand_expression, 1.0, 1.0, 1)
    nzind = numpy.where(alpha + beta != 0)
    S = numpy.zeros((len(adata_sasp), len(adata_t)))
    S[nzind] = (alpha * beta)[nzind] / (alpha + beta)[nzind]


    return S


def phi_exp(x, eta, nu, p):
    """The exponential weight kernel. Computes exp(-(x/eta)^(p*nu)).

    :param x: the input value
    :type x: float or class:`numpy.ndarray`
    :param eta: the cutoff for this soft thresholding kernel
    :type eta: float
    :param nu: a possitive integer for the power term, a bigger nu gives sharper threshold boundary
    :type nu: int
    :param p: p=1: emphasize elements lower than cutoff; p=-1: emphasize elements higher than cutoff
    :type p: int
    :return: the kernel output with same shape of x
    :rtype: same as x
    """
    epsilon = 1E-8
    y = numpy.empty_like(x)
    # if p == -1:
    #     nz_ind = numpy.where(x > epsilon)
    #     y = numpy.zeros_like(x)
    #     y[nz_ind] = numpy.exp(-numpy.power(x[nz_ind]/eta, p*nu))
    if p == -1:
        y = numpy.where(x > epsilon, numpy.exp(-numpy.power(x/eta, p*nu)), 0)
    else:
        # y = numpy.exp(-numpy.power(x/eta, p*nu))
        y = numpy.exp(-numpy.power(eta / x, p * nu))
    return y


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
    # always centralize on all cells, instead of included cells, if you want a different normalization, please change these codes.
    if centralize:
        background = y.mean(axis=1)
        y = y.subtract(background, axis=0)
    return y

def split_subsets(y):
    flag_group = [v.split('_')[1] for v in y.columns]
    result = y.groupby(flag_group, axis=1)
    return result

def interaction_test(expression, X, y, show_all_parameter=False):
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

        if show_all_parameter:
            tvalue_1 = result[2].loc['const'].iloc[0]
            tvalue_2 = result[2].loc['pivot'].iloc[0]
            tvalue_3 = result[2].loc['partner'].iloc[0]
            tvalue_4 = result[2].loc['interaction'].iloc[0]
            pvalue_1 = result[3].loc['const'].iloc[0]
            pvalue_2 = result[3].loc['pivot'].iloc[0]
            pvalue_3 = result[3].loc['partner'].iloc[0]
            pvalue_4 = result[3].loc['interaction'].iloc[0]
            merge.append(pandas.Series([tvalue_1, pvalue_1, tvalue_2, pvalue_2, tvalue_3, pvalue_3, tvalue_4, pvalue_4],
                                       index=['const', 'pivot', 'partner','interaction',
                                              'p_const', 'p_pivot', 'p_partner', 'p'], name=gid))
        else:
            tvalue = result[2].loc['interaction'].iloc[0]
            pvalue = result[3].loc['interaction'].iloc[0]
            merge.append(pandas.Series([tvalue, pvalue], index=['t', 'p'], name=gid))

    result = pandas.concat(merge, axis=1, join='inner').transpose()

    return result


def change_index(df):
    """
    #######
    :param df: 输入为gene*cell，在函数中转置,不需要提前转置
    :return: df
    """
    df = df.T
    v = df.groupby(df.index)\
        .cumcount()\
        .astype(str)\
        .values\
        .astype(str)
    df.index = (df.index.values + "."+v)
    df = df.T

    return df

