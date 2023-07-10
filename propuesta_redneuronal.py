import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from data_generation import gen_data

np.random.seed(69)
order = 10
ndata = 4000
nsample = 50

ntrain = round(0.6 * ndata)

x, sol, source = gen_data(order, ndata=ndata, nsample=nsample, sample_type='cheby')

sol = sol / np.max(np.abs(sol), axis=0)
source = source / np.max(np.abs(source), axis=0)

rand_idx = np.random.permutation(ndata)

x_train = source[rand_idx[:ntrain]]
y_train = sol[rand_idx[:ntrain]]
x_test = source[rand_idx[ntrain:]]
y_test = sol[rand_idx[ntrain:]]

solver = 'lbfgs'
alpha = 1e-5 
activation = 'identity'
hidden_layer_sizes = (10, 1)

mlp = MLPRegressor(solver=solver, alpha=alpha, activation=activation, hidden_layer_sizes=hidden_layer_sizes, random_state=69, max_iter=1000, verbose=True)
mlp.fit(x_train, y_train)

y_pred_train = mlp.predict(x_train)
y_pred_test = mlp.predict(x_test)

plt.figure(figsize=(10, 5))
plt.plot(x, y_train[0], label='True Solution (Train)')
plt.plot(x, y_pred_train[0], label='Predicted Solution (Train)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(x, y_test[0], label='True Solution (Test)')
plt.plot(x, y_pred_test[0], label='Predicted Solution (Test)')
plt.legend()
plt.show()