import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from data_generation import gen_data

np.random.seed(69)

order = 10
ndata = 4000
nsample = 50

x, sol, source = gen_data(order, ndata=ndata, nsample=nsample, sample_type="cheby")

sol = sol / np.max(np.abs(sol), axis=0)
source = source / np.max(np.abs(source), axis=0)

solver = 'lbfgs'
alpha = 1e-5
activation = 'identity'
hidden_layer_sizes = (10, 1)

mlp = MLPRegressor(solver=solver, alpha=alpha, activation=activation, hidden_layer_sizes=hidden_layer_sizes, random_state=69, max_iter=1000)

k = 5
cv_scores = cross_val_score(mlp, source, sol, cv=k, scoring='r2')

results = pd.DataFrame({
    "Fold": np.arange(1, k + 1),
    "R^2 Score": cv_scores
})

results.loc['Mean'] = results.mean(numeric_only=True)
results.loc['Standard Deviation'] = results.std(numeric_only=True)

results.to_csv("cross_validation_results.csv", index=False)