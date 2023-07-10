#!/usr/bin/env python
# coding: utf-8

# # Visualización exploratoria

# ### Importar librerías
# 
# Se importan los datos de la solución de la ecuación de Poisson

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import *
from data_generation import gen_data
import seaborn as sns

# Directorio para guardar las gráficas
dir = 'C:\\Users\\Jose Manuel\\OneDrive - Universidad EAFIT\\ml_pde\\codigo\\jose\\obtencion_matriz\\visualizacion\\img\\'

sns.set()
np.random.seed(69)


# Se parten los datos para entrenamiento y prueba (también se normalizan)

# In[2]:


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


# ## Arquitectura de la red neuronal

# In[3]:


solver = 'lbfgs'
alpha = 1e-5 # Término de regularización para la red neuronal
activation = 'identity'
hidden_layer_sizes = (10, 1)

clf = MLPRegressor(solver=solver, alpha=alpha, activation=activation, hidden_layer_sizes=hidden_layer_sizes, random_state=69, max_iter=1000, verbose=True)
clf.fit(x_train, y_train)


# In[4]:


def plot_scatter_comparison(x, y, xlabel, ylabel, title, label1, label2, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, 'o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend([label1, label2])


# Visualización exploratoria de los datos
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

plot_scatter_comparison(x_train[:, 0], y_train[:, 0], 'x', 'y', 'Training set', 'Train', None, axs[0, 0])
plot_scatter_comparison(x_test[:, 0], y_test[:, 0], 'x', 'y', 'Testing set', 'Test', None, axs[0, 1])

# Evaluación del desempeño de la red
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)

print(f"Train score: {train_score}")
print(f"Test score: {test_score}")

plot_scatter_comparison(y_train[:, 0], y_pred_train[:, 0], 'True values', 'Predicted values', 'Training set', 'True', 'Predicted', axs[1, 0])
plot_scatter_comparison(y_test[:, 0], y_pred_test[:, 0], 'True values', 'Predicted values', 'Testing set', 'True', 'Predicted', axs[1, 1])

plt.tight_layout()


plt.savefig(dir + 'vis_Gráficos de dispersión para explorar los datos.pdf')


# # Visualización exploratoria

# ## Exploración de los datos

# In[5]:


# Visualización exploratoria de los datos
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(x_train[:, 0], y_train[:, 0], 'o', label='Train')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')
axs[0, 0].set_title('Training set')
axs[0, 0].legend()

axs[0, 1].plot(x_test[:, 0], y_test[:, 0], 'o', label='Test')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')
axs[0, 1].set_title('Testing set')
axs[0, 1].legend()

# Evaluación del desempeño de la red
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)

print(f"Train score: {train_score}")
print(f"Test score: {test_score}")

axs[1, 0].plot(y_train[:, 0], y_pred_train[:, 0], 'o')
axs[1, 0].set_xlabel('True values')
axs[1, 0].set_ylabel('Predicted values')
axs[1, 0].set_title('Training set')

axs[1, 1].plot(y_test[:, 0], y_pred_test[:, 0], 'o')
axs[1, 1].set_xlabel('True values')
axs[1, 1].set_ylabel('Predicted values')
axs[1, 1].set_title('Testing set')

plt.tight_layout()


# ### Comparación de la solución verdadera y la solución aproximada por la red neuronal en un solo punto: 
# 
# Esta gráfica muestra la comparación entre la solución verdadera de la ecuación diferencial de Poisson y la solución aproximada por la red neuronal en un solo punto.

# In[6]:


y_true = sol[0]
y_pred = clf.predict(source[0].reshape(1, -1))

plt.figure()
plt.plot(y_true, label='Solución verdadera')
plt.plot(y_pred[0], label='Solución aproximada')
plt.xlabel('Índice de muestra')
plt.ylabel('Valor de la solución')
plt.title('Comparación de la solución verdadera y la solución aproximada por la red neuronal en un punto específico', fontsize=9.5)
plt.legend()
plt.savefig(dir+ 'vis_Comparación de la solución verdadera y la solución aproximada por la red neuronal en un punto específico.pdf')
plt.show()


# ### Evolución temporal de la solución verdadera y la solución aproximada por la red neuronal: 
# 
# Esta gráfica muestra cómo evoluciona temporalmente la solución verdadera de la ecuación diferencial de Poisson y la solución aproximada por la red neuronal.

# In[7]:


y_true = sol
y_pred = clf.predict(source)

plt.figure()
plt.plot(y_true[:, 0], label='Solución verdadera')
plt.plot(y_pred[:, 0], label='Solución aproximada')
plt.xlabel('Índice de muestra')
plt.ylabel('Valor de la solución')
plt.title('Evolución temporal de la solución verdadera y la solución aproximada por la red neuronal', fontsize=11.3)
plt.legend()
plt.savefig(dir+ 'vis_Evolución temporal de la solución verdadera y la solución aproximada por la red neuronal.pdf')
plt.show()


# ### Distribución de los errores en la solución aproximada por la red neuronal: 
# 
# Esta gráfica muestra la distribución de los errores entre la solución verdadera de la ecuación diferencial de Poisson y la solución aproximada por la red neuronal en los datos de prueba.

# In[8]:


errors = y_test - clf.predict(x_test)
plt.figure()
sns.kdeplot(errors.flatten())
plt.xlabel('Valor del error')
plt.ylabel('Densidad de probabilidad')
plt.title('Distribución de los errores en la solución aproximada por la red neuronal')
plt.savefig(dir+ 'vis_Distribución de los errores en la solución aproximada por la red neuronal.pdf')
plt.show()


# ### Evolución temporal de los errores en la solución aproximada por la red neuronal en el conjunto de prueba: 
# 
# Esta gráfica muestra cómo evoluciona temporalmente los errores entre la solución verdadera de la ecuación diferencial de Poisson y la solución aproximada por la red neuronal.

# In[9]:


y_true = y_test
y_pred = clf.predict(x_test)

errors = y_true - y_pred

plt.figure()
plt.plot(errors)
plt.xlabel('Índice de muestra')
plt.ylabel('Valor del error')
plt.title('Evolución temporal de los errores en la solución aproximada por la red neuronal en el conjunto de prueba', fontsize=9)
plt.savefig(dir+ 'vis_Evolución temporal de los errores en la solución aproximada por la red neuronal en el conjunto de prueba.pdf')
plt.show()


# ### Evolución de la función de costo durante el entrenamiento de la red neuronal: 
# 
# Esta gráfica muestra cómo evoluciona la función de costo durante el entrenamiento de la red neuronal.

# In[10]:


# Calcular las pérdidas

losses = []
for i in range(100):
    clf.fit(x_train, y_train)
    loss = clf.loss_
    losses.append(loss)

# Visualizar la curva de pérdidas

plt.figure()
plt.plot(losses)
plt.xlabel('Número de iteraciones')
plt.ylabel('Valor de la función de costo')
plt.title('Evolución de la función de costo durante el entrenamiento de la red neuronal')
plt.savefig(dir+ 'vis_Evolución de la función de costo durante el entrenamiento de la red neuronal.pdf')
plt.show()


# ### Evolución de la tasa de aprendizaje durante el entrenamiento de la red neuronal: 
# 
# Esta gráfica muestra cómo evoluciona la tasa de aprendizaje durante el entrenamiento de la red neuronal.

# In[11]:


plt.figure()
plt.plot(clf.learning_rate_init - np.arange(clf.n_iter_) * clf.learning_rate_init / clf.n_iter_)
plt.xlabel('Número de iteraciones')
plt.ylabel('Tasa de aprendizaje')
plt.title('Evolución de la tasa de aprendizaje durante el entrenamiento de la red neuronal')
plt.savefig(dir+ 'vis_Evolución de la tasa de aprendizaje durante el entrenamiento de la red neuronal.pdf')
plt.show()


# ### Eficiencia de la red neuronal en función de la complejidad de la arquitectura: 
# 
# Esta gráfica muestra cómo la eficiencia de la red neuronal en la solución de la ecuación diferencial de Poisson cambia con la complejidad de la arquitectura.

# In[12]:


n_neurons = [2, 5, 10, 20, 50, 100]
scores = []

for n in n_neurons:
    clf = MLPRegressor(solver=solver, alpha=alpha, activation=activation, hidden_layer_sizes=(n, 1), random_state=69, max_iter=1000)
    clf.fit(x_train, y_train)
    scores.append(clf.score(x_test, y_test))

plt.figure()
plt.plot(n_neurons, scores, '-o')
plt.xlabel('Número de neuronas en la capa oculta')
plt.ylabel('Puntaje de prueba')
plt.title('Eficiencia de la red neuronal en función de la complejidad de la arquitectura')
plt.savefig(dir+ 'vis_Eficiencia de la red neuronal en función de la complejidad de la arquitectura.pdf')
plt.show()


# ### Solución verdadera vs solución aproximada:
# 
# Esta gráfica muestra cómo la solución verdadera de la ecuación diferencial de Poisson y la solución aproximada por la red neuronal se relacionan en el espacio de fase.

# In[13]:


plt.figure()
plt.plot(sol[:, 0], sol[:, 1], label='Solución verdadera')
plt.plot(y_pred[:, 0], y_pred[:, 1], label='Solución aproximada')
plt.xlabel('Valor de la solución en x')
plt.ylabel('Valor de la solución en y')
plt.title('Solución verdadera vs solución aproximada')
plt.legend()
plt.show()


# ### Distribución de los errores en la solución aproximada por la red neuronal por capa:
# 
# Esta gráfica muestra la distribución de los errores entre la solución verdadera de la ecuación diferencial de Poisson y la solución aproximada por la red neuronal en los datos de prueba, desglosados por capa de la red neuronal.

# In[36]:


y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

errors_train = y_train - y_pred_train
errors_test = y_test - y_pred_test

n_layers = clf.n_layers_

fig, axs = plt.subplots(1, n_layers - 1, figsize=(15, 5))

for i, ax in enumerate(axs.flatten()):
    if i == 0:
        errors = errors_test
        title = 'Errores en datos de prueba'
    else:
        errors = clf.coefs_[i - 1]
        title = f'Capa {i}'

    if np.var(errors.flatten()) != 0:
        sns.kdeplot(errors.flatten(), ax=ax)
        ax.set_xlabel('Valor del error')
        ax.set_ylabel('Densidad de probabilidad')
        ax.set_title(title)

plt.suptitle('Distribución de los errores en la solución aproximada por la red neuronal por capa')
plt.tight_layout()
plt.savefig(dir+ 'vis_Distribución de los errores en la solución aproximada por la red neuronal por capa.pdf')
plt.show()


# ### Histograma de errores: 
# 
# Muestra la distribución de los errores entre la solución verdadera y la solución predicha por la red neuronal. 
# 
# Distribución simétrica alrededor de cero: la red está bien ajustada.

# In[15]:


y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

errors_train = y_train - y_pred_train
errors_test = y_test - y_pred_test

plt.figure(figsize=(8, 8))
plt.hist(errors_train, bins=50, alpha=0.5, label='Errores en datos de entrenamiento')
plt.hist(errors_test, bins=50, alpha=0.5, label='Errores en datos de prueba')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
plt.title('Histograma de errores en la solución aproximada por la red neuronal')
plt.legend()
plt.show()


# ### Curva de aprendizaje: 
# 
# Muestra cómo cambia la precisión de la red neuronal a medida que se aumenta el tamaño del conjunto de entrenamiento.
# 
# Para el ajuste de la red, la precisión debe mejorar a medida que se aumenta el tamaño del conjunto de entrenamiento.

# In[16]:


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(clf, x_train, y_train, cv=5)

plt.figure(figsize=(8, 8))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Precisión en datos de entrenamiento')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Precisión en datos de prueba')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Precisión')
plt.title('Curva de aprendizaje de la red neuronal')
plt.legend()
plt.show()


# ### Gráfica de residuos: 
# 
# Compara los residuos (los errores) entre la solución real y la solución predicha por la red neuronal en una nube de puntos. 
# 
# Para el ajuste de la red neuronal, los residuos deben estar distribuidos aleatoriamente alrededor de cero.

# In[17]:


y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

errors_train = y_train - y_pred_train
errors_test = y_test - y_pred_test

plt.figure(figsize=(8, 8))
plt.scatter(y_pred_train[:, 0], errors_train[:, 0], label='Datos de entrenamiento')
plt.scatter(y_pred_test[:, 0], errors_test[:, 0], label='Datos de prueba')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Solución predicha')
plt.ylabel('Residuo')
plt.title('Gráfica de residuos de la solución aproximada por la red neuronal')
plt.legend()
plt.show()


# ### Graficar la distribución de los errores absolutos entre la solución real y la solución aproximada
# 
# Se grafica la distribución de los errores absolutos entre la solución real y la solución aproximada para todos los puntos de prueba.

# In[18]:


errors = np.abs(y_test - clf.predict(x_test))
plt.figure(figsize=(8, 5))
plt.hist(errors.ravel(), bins=50, density=True)
plt.xlabel('Error absoluto')
plt.ylabel('Frecuencia')
plt.title('Distribución de los errores absolutos')
plt.show()


# ### Graficar la evolución del puntaje de la red neuronal durante el entrenamiento
# 
# Se grafica la evolución del puntaje de la red neuronal durante el entrenamiento, lo que permitiría tener una mejor idea de cómo se desempeña la red neuronal durante el aprendizaje:

# In[19]:


# Graficar la evolución del puntaje de la red neuronal durante el entrenamiento
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, activation='identity', hidden_layer_sizes=(10, 1), max_iter=1000, warm_start=True)

scores = []
tol = 1e-4
n_iter_no_change = 10

for i in range(100):
    clf.fit(x_train, y_train)
    score = clf.score(x_test,y_test)
    scores.append(score)
    if i > n_iter_no_change and np.abs(scores[-1] - scores[-(n_iter_no_change+1)]) < tol:
        break

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(scores)+1), scores)
plt.xlabel('Número de iteraciones')
plt.ylabel('Puntaje')
plt.title('Evolución del puntaje durante el entrenamiento')
plt.show()


# ### Graficar la solución real y la solución aproximada para diferentes órdenes de la ecuación diferencial
# 
# Se grafica la solución real y la solución aproximada para diferentes órdenes de la ecuación diferencial para tener una mejor idea de cómo se desempeña la red neuronal para diferentes grados de complejidad de la ecuación. 

# In[20]:


orders = [2, 5, 10, 20, 50]
fig, axs = plt.subplots(len(orders), 1, figsize=(8, 15), sharex=True, sharey=True)
for i, o in enumerate(orders):
    x, sol, source = gen_data(o, ndata=ndata, nsample=nsample, sample_type="cheby")
    sol, source = sol / np.max(np.abs(sol), axis=0), source / np.max(np.abs(source), axis=0)
    x_train, y_train = source[rand_idx[:ntrain]], sol[rand_idx[:ntrain]]
    x_test, y_test = source[rand_idx[ntrain:]], sol[rand_idx[ntrain:]]
    clf = MLPRegressor(solver='lbfgs', alpha=1e-5, activation='identity', hidden_layer_sizes=(10, 1), random_state=69, max_iter=1000)
    clf.fit(x_train, y_train)
    axs[i].plot(x, y_test[0,:], label='Solución real')
    axs[i].plot(x, clf.predict(x_test)[0,:], label='Solución aproximada')
    axs[i].set_title(f"order = {o}")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.suptitle('Comparación de la solución real y la solución aproximada para diferentes órdenes de la ecuación diferencial')
plt.show()


# ### Graficar la evolución del error absoluto promedio entre la solución real y la solución aproximada
# 
# Se grafica la evolución del error absoluto promedio entre la solución real y la solución aproximada a lo largo del entrenamiento, para tener una idea de cómo el error disminuye a medida que la red neuronal aprende:

# In[21]:


clf = MLPRegressor(solver='lbfgs', alpha=1e-5, activation='identity', hidden_layer_sizes=(10, 1), random_state=69, max_iter=1000, warm_start=True)

errors = []
for i in range(100):
    clf.fit(x_train, y_train)
    error = np.mean(np.abs(y_test - clf.predict(x_test)))
    errors.append(error)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(errors)+1), errors)
plt.xlabel('Número de iteraciones')
plt.ylabel('Error absoluto promedio')
plt.title('Evolución del error absoluto promedio durante el entrenamiento')
plt.show()


# ### Graficar la distribución del error absoluto entre la solución real y la solución aproximada
# 
# Se grafica la distribución del error absoluto entre la solución real y la solución aproximada para cada punto de prueba, para tener una idea de la variabilidad del error a lo largo de los diferentes puntos de prueba:

# In[22]:


errors = np.abs(y_test - clf.predict(x_test))
plt.figure(figsize=(8, 5))
plt.boxplot(errors)
plt.xlabel('Puntos de prueba')
plt.ylabel('Error absoluto')
plt.title('Distribución del error absoluto en los diferentes puntos de prueba')
plt.show()


# ### Gráfica de violines: 
# 
# Muestra la distribución del error absoluto en los diferentes puntos de prueba mediante una serie de violines. Cada violín representa una distribución de frecuencia del error absoluto en un punto de prueba específico. El tamaño del violín representa la densidad de probabilidad en esa ubicación y los puntos blancos dentro de los violines representan la mediana del error absoluto en esa ubicación. Si la red neuronal está bien ajustada, los violines deben ser simétricos alrededor de la mediana y tener un tamaño similar en cada ubicación.

# In[23]:


errors = np.abs(y_test - clf.predict(x_test))
plt.figure(figsize=(8, 5))
plt.violinplot(errors)
plt.xlabel('Puntos de prueba')
plt.ylabel('Error absoluto')
plt.title('Distribución del error absoluto en los diferentes puntos de prueba')
plt.show()


# ### Heatmap de los pesos de la red neuronal: 
# 
# Esta visualización muestra los pesos de las conexiones en la red neuronal en forma de un mapa de calor. Los valores numéricos indican la fuerza de la conexión entre las neuronas de cada capa, con colores rojos para valores positivos y azules para valores negativos.

# In[24]:


# Obtener los pesos de la red neuronal
weights_1 = clf.coefs_[0]
weights_2 = clf.coefs_[1]

# Graficar los pesos de la primera capa oculta como un heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(weights_1, cmap='coolwarm', annot=True)
plt.title('Pesos de la primera capa oculta')
plt.xlabel('Neuronas en la capa oculta')
plt.ylabel('Variables de entrada')
plt.show()

# Graficar los pesos de la segunda capa oculta como un heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(weights_2, cmap='coolwarm', annot=True)
plt.title('Pesos de la segunda capa oculta')
plt.xlabel('Neuronas en la capa oculta')
plt.ylabel('Neuronas en la capa anterior')
plt.show()

