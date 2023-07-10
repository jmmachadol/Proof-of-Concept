import numpy as np
import pandas as pd
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from data_generation import gen_data_mms
import warnings

warnings.filterwarnings("ignore")

# Generar datos
x, sol, source = gen_data_mms()

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(source, sol, test_size=0.3, random_state=42)

# Escalar los datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Establecer el rango de hiperparámetros
param_dist = {
    "hidden_layer_sizes": [(a,) for a in range(10, 101, 10)],
    "activation": ["identity", "logistic", "tanh", "relu"],
    "solver": ["lbfgs", "sgd", "adam"],
    "alpha": np.logspace(-5, 0, num=10),
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "learning_rate_init": np.logspace(-5, 0, num=10),
    "max_iter": [100, 200, 300, 400, 500],
    "batch_size": ["auto"] + [i for i in range(10, 101, 10)],
}

# Crear el modelo base
mlp = MLPRegressor(random_state=42)

# Crear la búsqueda aleatoria
random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, random_state=42)

# Medir el tiempo de cómputo
start_time = time.time()

# Ajustar la búsqueda aleatoria a los datos de entrenamiento
random_search.fit(X_train, y_train)

# Calcular el tiempo de cómputo
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo de cómputo: {elapsed_time} segundos")

# Imprimir los mejores hiperparámetros encontrados
print("Best hyperparameters found: ", random_search.best_params_)

# Crear y entrenar el modelo final con los mejores hiperparámetros
best_mlp = random_search.best_estimator_
best_mlp.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo en el conjunto de prueba
y_pred = best_mlp.predict(X_test)

# Imprimir un DataFrame con el desempeño de la red para cada combinación de hiperparámetros
results_df = pd.DataFrame(random_search.cv_results_)
print("Random search results:")
print(results_df)
# Guardar los resultados en un archivo CSV
results_df.to_csv('random_search_results.csv', index=False)