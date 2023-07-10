import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from data_generation import gen_data_mms

# Función para ensamblar la matriz de rigidez
def assemble_stiffness_matrix(x):
    n = len(x) - 1
    h = x[1] - x[0]
    diagonals = np.zeros((3, n+1))
    diagonals[0, 1:] = -1/h
    diagonals[1] = 2/h
    diagonals[2, :-1] = -1/h
    K = diags(diagonals, [-1, 0, 1], shape=(n+1, n+1), format='csr')
    return K

# Función para ensamblar el vector de carga
def assemble_load_vector(x, f):
    n = len(x) - 1
    b = np.zeros(n+1)
    for i in range(n):
        h = x[i+1] - x[i]
        b[i] += h/2*f(x[i])
        b[i+1] += h/2*f(x[i+1])
    return b

# Función para aplicar las condiciones de frontera
def apply_boundary_conditions(K, b, bc):
    ua, ub = bc
    K[0,:] = 0
    K[0,0] = 1
    b[0] = ua
    K[-1,:] = 0
    K[-1,-1] = 1
    b[-1] = ub
    return K, b

# Generar datos utilizando la función gen_data_mms de data_generation.py
x, sol, source = gen_data_mms(ndata=50)

# Configurar malla FEM
x_mesh = np.linspace(0, 1, 50)

# Lista para almacenar errores
errors = []

# Iterar a través de los conjuntos de datos generados
for i in range(source.shape[0]):
    source_function = source[i, :]
    bc = (sol[i, 0], sol[i, -1])

    # Ensamblar la matriz de rigidez y el vector de carga
    K = assemble_stiffness_matrix(x_mesh)
    b = assemble_load_vector(x_mesh, lambda x_value: np.interp(x_value, x, source_function))

    # Aplicar condiciones de frontera
    K, b = apply_boundary_conditions(K, b, bc)

    # Resolver el sistema de ecuaciones utilizando un solver de matrices dispersas
    u_fem = spsolve(K, b)

    # Comparar la solución FEM con la solución analítica
    u_analytical = np.interp(x_mesh, x, sol[i, :])  # Interpolación de la solución analítica en x_mesh
    error = np.linalg.norm(u_fem - u_analytical) / np.linalg.norm(u_analytical)
    errors.append(error)

# Calcular estadísticas de error
errors = np.array(errors)
print("Error relativo promedio:", np.mean(errors))
print("Desviación estándar del error relativo:", np.std(errors))

#Graficar la solución FEM y la solución analítica para un caso de prueba
plt.plot(x_mesh, u_fem, label='FEM')
plt.plot(x_mesh, u_analytical, label='Analítica') # Usamos u_analytical interpolada en x_mesh
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()