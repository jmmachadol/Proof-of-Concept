import numpy as np
from data_generation import gen_data

# Configuraciones de conjuntos de datos
data_configs = [
    {'n': 10, 'ndata': 2000, 'nsample': 50, 'sample_type': 'uniform'},
    {'n': 20, 'ndata': 3000, 'nsample': 100, 'sample_type': 'uniform'},
    {'n': 30, 'ndata': 4000, 'nsample': 150, 'sample_type': 'uniform'},
    {'n': 10, 'ndata': 2000, 'nsample': 50, 'sample_type': 'cheby'},
    {'n': 20, 'ndata': 3000, 'nsample': 100, 'sample_type': 'cheby'},
    {'n': 30, 'ndata': 4000, 'nsample': 150, 'sample_type': 'cheby'}
]

data_sets = []

# Generar conjuntos de datos
for config in data_configs:
    x, sol, source = gen_data(n=config['n'], ndata=config['ndata'], nsample=config['nsample'], sample_type=config['sample_type'])
    data_sets.append((x, sol, source))

# Normalizar los datos
for i, (x, sol, source) in enumerate(data_sets):
    sol = sol / np.max(np.abs(sol), axis=0)
    source = source / np.max(np.abs(source), axis=0)
    data_sets[i] = (x, sol, source)