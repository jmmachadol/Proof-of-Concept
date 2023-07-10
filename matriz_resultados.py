import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import *
from data_generation import gen_data
import seaborn as sns

sns.set()

np.random.seed(69)

# Parámetros y generación de datos
order = 10
ndata = 4000
nsample = 50
ntrain = round(0.6 * ndata)

x, sol, source = gen_data(order, ndata=ndata, nsample=nsample, sample_type="cheby")
sol = sol / np.max(np.abs(sol), axis=0)
source = source / np.max(np.abs(source), axis=0)

rand_idx = np.random.permutation(ndata)
x_train = source[rand_idx[:ntrain]]
y_train = sol[rand_idx[:ntrain]]
x_test = source[rand_idx[ntrain:]]
y_test = sol[rand_idx[ntrain:]]

# Configuración y entrenamiento de la red neuronal
solver = 'lbfgs'
alpha = 1e-5
activation = 'identity'
hidden_layer_sizes = (10, 1)

mlp = MLPRegressor(solver=solver, alpha=alpha, activation=activation, hidden_layer_sizes=hidden_layer_sizes, random_state=69, max_iter=1000, verbose=True)
mlp.fit(x_train, y_train)

test_score = mlp.score(x_test, y_test)
print(f"Test score: {test_score}")

# Funciones para el análisis de la matriz y las visualizaciones
def compute_matrix(mlp, nsample):
    identity = np.eye(nsample)
    matrix = np.zeros_like(identity)
    col = np.zeros(nsample)
    col.shape = 1, nsample
    offset = mlp.predict(col)
    offset.shape = nsample
    
    for cont in range(nsample):
        col = identity[:, cont]
        col.shape = 1, nsample
        row = mlp.predict(col) - offset
        matrix[cont, :] = row
    
    return matrix, offset

def decompose_matrix(matrix):
    mat_sym = 0.5 * (matrix + matrix.T)
    mat_skew = 0.5 * (matrix - matrix.T)
    return mat_sym, mat_skew

def plot_matrices(matrix, mat_sym, mat_skew):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axs[0].imshow(matrix, cmap='Spectral')
    axs[0].set_title('Matrix')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Index')
    axs[0].grid(False)
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(mat_sym)
    axs[1].set_title('Symmetric Component')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Index')
    axs[1].grid(False)
    fig.colorbar(im2, ax=axs[1])

    im3 = axs[2].imshow(mat_skew)
    axs[2].set_title('Skew-symmetric Component')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel('Index')
    axs[2].grid(False)
    fig.colorbar(im3, ax=axs[2])
    plt.tight_layout()
    plt.savefig('matrix_visualization.pdf', bbox_inches='tight')
    plt.show()

def compare_predictions(mlp, matrix, offset, nsample):
    vec = np.random.normal(0, 1, (1, nsample))
    u1 = mlp.predict(vec)
    u2 = matrix.T @ vec.flatten() + offset
    plt.figure()
    plt.plot(u1.flatten(), lw=5, label='Neural Network Prediction')
    plt.plot(u2, label='Matrix-based Prediction')
    plt.xlabel('Index')
    plt.ylabel('Prediction Value')
    plt.legend()
    plt.savefig('predictions_comparison.pdf', bbox_inches='tight')
    plt.show()

# Análisis de la matriz y visualización
matrix, offset = compute_matrix(mlp, nsample)
mat_sym, mat_skew = decompose_matrix(matrix)
plot_matrices(matrix, mat_sym, mat_skew)
compare_predictions(mlp, matrix, offset, nsample)