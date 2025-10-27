import os
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ridge_regression, LinearRegression


def linear_interaction_input(x):
    cy = x.loc[:, [c for c in x.columns if c.startswith('cy')]]
    genes = x.loc[:, [c for c in x.columns if c.startswith('ge')]]
    cy_value = cy.iloc[:, 0]
    for gene in genes.columns:
        values = genes.loc[:, gene]
        x.loc[:, f'Interaction_{cy.columns.tolist()[0].split("_")[1]}_{gene.split("_")[1]}'] = cy_value * values
    return x

organ = 'skin'
cell_type = 'T'
output_dir = r'model path/att_map'
dataPath = fr'data_demo/senescenc/{organ}/shap/input/{cell_type}'
if not os.path.exists(resultPath):
    os.makedirs(resultPath)

cy_list = list(set([f.split('_')[0] for f in os.listdir(dataPath)]))
cy_list.sort()

param_grid = {'C': [1]}
svr = SVR(kernel='linear')
grid_search = GridSearchCV(svr, param_grid, n_jobs=-1)
models = {'MLP': MLPRegressor(hidden_layer_sizes=(100, 100), solver="lbfgs", alpha=1e-2, random_state=42),
          'SVR': grid_search,
          'RF': RandomForestRegressor(random_state=42),
          'LR': LinearRegression(n_jobs=-1)}

model_list = list(models.keys())
for model_name in model_list:
    resultPath = fr'data_demo/senescenc/{organ}/shap/result/{model_name}/{cell_type}'
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    Xdata = []
    for cy in cy_list:
        # Read the data
        df = pd.read_csv(os.path.join(dataPath, f"{cy}_input.csv"), index_col=False)
        df.set_index(df.iloc[:, 0], inplace=True)
        df.dropna(axis=1, inplace=True)
        X = df.drop(columns=[f'activate', 'Unnamed: 0'])
        Xdata.append(X)
        y = df[f'activate']
    Xdata = pd.concat(Xdata, axis=1)
    Xdata.sort_index(axis=1, ascending=True, inplace=True)
    Xdata = Xdata.loc[:, ~Xdata.columns.duplicated()]
    X = Xdata
    fetures_num = len(Xdata.columns)
    interaction = pd.read_csv(os.path.join(output_dir, f'att_map_{cell_type}-Cell_Top20InMap.csv'))
    interactions = interaction['cy'].astype(str) + '~' + interaction['gene']

    X = linear_interaction_input(X, list(interactions))
    Y = y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    regressor = models[model_name]
    regressor.fit(X_train, y_train)


    X_train_summary = shap.kmeans(X_train, 10)
    explainer = shap.Explainer(regressor, X_train)
    shap_values = explainer(X)
    shap_interaction_values = shap_values[:, fetures_num:]
    # Identify column indices for 'cytosig_' and 'gene_'
    cytosig_indices = [i for i, col in enumerate(X.columns) if col.startswith('cytosig_')]
    gene_indices = [i for i, col in enumerate(X.columns) if col.startswith('gene_')]

    all_interaction_values_df = pd.DataFrame(columns=X.columns[fetures_num:], data=shap_interaction_values)

    all_interaction_values_df.to_csv(os.path.join(resultPath, f"shap_interaction_FULL.csv"), index=False)

    import pickle

    with open(os.path.join(resultPath, f"explainer.pkl"), 'wb') as f:
        pickle.dump(explainer, f)