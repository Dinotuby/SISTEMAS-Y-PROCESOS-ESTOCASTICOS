import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --------------------------
# a) Generar N variables iid con parámetros de la distribución geométrica
# --------------------------
p = 0.3              # Parámetro de la distribución geométrica (probabilidad de éxito)
N = 1000             # Cantidad de variables aleatorias iid (columnas)
n = 10000            # Cantidad de realizaciones por variable (filas)

# Generamos una matriz de tamaño n x N donde cada columna representa una variable aleatoria
X = np.random.geometric(p, size=(n, N))

# --------------------------
# c) Sumar a lo largo de las columnas (es decir, sumar por fila)
# Resultado: vector con n sumas
# --------------------------
Z_c = np.sum(X, axis=1)

# Media y desviación estándar teórica para la suma
media_c = N * (1 / p)
std_c = np.sqrt(N * (1 - p) / p**2)

# d) Histograma de las sumas y comparación con la distribución normal
plt.figure(figsize=(10, 6))
plt.hist(Z_c, bins=50, density=True, alpha=0.6, label="Histograma Z (sumas por fila)")
x_vals = np.linspace(min(Z_c), max(Z_c), 1000)
plt.plot(x_vals, norm.pdf(x_vals, media_c, std_c), 'r', label="Normal teórica")
plt.title("c-d) Suma de variables iid (por fila, n sumas)")
plt.xlabel("Valor de la suma")
plt.ylabel("Densidad")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# e) Sumar a lo largo de las filas (es decir, sumar por columna)
# Resultado: vector con N sumas
# --------------------------
Z_e = np.sum(X, axis=0)

# Media y desviación estándar teórica para esta suma
media_e = n * (1 / p)
std_e = np.sqrt(n * (1 - p) / p**2)


# Distribución de los promedios
promedios = Z_e / n  # n = 10_000

# Histograma de las sumas por columna y comparación con normal


plt.figure(figsize=(10, 6))
plt.hist(Z_e, bins=50, density=True, alpha=0.6, label="Histograma Z (sumas por columna)")
x_vals2 = np.linspace(min(Z_e), max(Z_e), 1000)
plt.plot(x_vals2, norm.pdf(x_vals2, media_e, std_e), 'r', label="Normal teórica")
plt.title("e) Suma de realizaciones iid (por columna, N sumas)")
plt.xlabel("Valor de la suma")
plt.ylabel("Densidad")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Tomar solo la primera columna
X1 = X[:, 0]

# Calcular el promedio acumulado: cómo varía el promedio a medida que sumamos más datos
prom_acumulado = np.cumsum(X1) / np.arange(1, n + 1)

# Valor esperado teórico de la geométrica
valor_esperado = 1 / p

# Graficar
plt.figure(figsize=(10, 5))
plt.plot(prom_acumulado, label='Promedio acumulado (columna 1)', color='purple')
plt.axhline(valor_esperado, color='orange', linestyle='--', label=f'Valor esperado = {valor_esperado:.2f}')
plt.xlabel("Número de observaciones")
plt.ylabel("Promedio acumulado")
plt.title("Convergencia del promedio de una variable aleatoria (LGN)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
