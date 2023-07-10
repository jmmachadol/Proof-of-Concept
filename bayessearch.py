import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from data_generation import gen_data_mms

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

# Establecer el espacio de hiperparámetros
param_space = {
    "hidden_layer_sizes": Integer(10, 100),
    "activation": Categorical(["identity", "logistic", "tanh", "relu"]),
    "solver": Categorical(["lbfgs", "sgd", "adam"]),
    "alpha": Real(1e-5, 1, prior="log-uniform"),
    "learning_rate": Categorical(["constant", "invscaling", "adaptive"]),
    "learning_rate_init": Real(1e-5, 1, prior="log-uniform"),
    "max_iter": Integer(100, 500),
    "batch_size": Integer(10, 100),
}

# Crear el modelo base
mlp = MLPRegressor(random_state=42)

# Crear la búsqueda bayesiana
bayes_search = BayesSearchCV(mlp, search_spaces=param_space, n_iter=50, cv=3, n_jobs=-1, random_state=42)

# Medir el tiempo de cómputo
start_time = time.time()

# Ajustar la búsqueda bayesiana a los datos de entrenamiento
bayes_search.fit(X_train_scaled, y_train_scaled)

# Calcular el tiempo de cómputo
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo de cómputo: {elapsed_time} segundos")

# Imprimir los mejores hiperparámetros encontrados
print("Best hyperparameters found: ", bayes_search.best_params_)

# Crear y entrenar el modelo final con los mejores hiperparámetros
best_mlp = bayes_search.best_estimator_
best_mlp.fit(X_train_scaled, y_train_scaled)

# Realizar predicciones y evaluar el modelo en el conjunto de prueba
y_pred_scaled = best_mlp.predict(X_test_scaled)

# Transformar las predicciones a la escala original
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Imprimir un DataFrame con el desempeño de la red para cada combinación de hiperparámetros
results_df = pd.DataFrame(bayes_search.cv_results_)
print("Bayesian optimization results:")
print(results_df)

# Guardar los resultados en un archivo CSV
results_df.to_csv('bayes_search_results.csv', index=False)