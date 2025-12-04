"""
Script: cluster.py

Descripción:
Este script carga las matrices extrínsecas generadas por un modelo de visión por computadora (como VGGT),
calcula las posiciones de la cámara en el espacio 3D y aplica clustering usando K-Means para agrupar
estas posiciones. Finalmente, visualiza los clusters en un gráfico 3D.

Requisitos:
- Python 3.7 o superior
- Librerías instaladas:
    - pandas
    - numpy
    - opencv-python
    - matplotlib

"""

import pandas as pd
import numpy as np
import cv2
import ast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necesario para gráficos 3D en matplotlib

# Cargar archivo CSV generado por otro script, que contiene las matrices extrínsecas de las cámaras
df = pd.read_csv("C:\\Users\\itsal\\Desktop\\xd\\data.csv")

positions = []

# Procesar cada fila para calcular la posición de la cámara a partir de la matriz extrínseca
for extrinsic_str in df["extrinsic"]:
    # Convertir el string de la lista en un array de numpy
    extrinsic = np.array(ast.literal_eval(extrinsic_str)).reshape((3, 4))

    # Separar la matriz de rotación (R) y el vector de traslación (T)
    R = extrinsic[:, :3]
    T = extrinsic[:, 3]

    # Calcular la posición de la cámara en coordenadas mundiales: C = -R^T * T
    cam_position = -R.T @ T
    positions.append(cam_position)

# Convertir la lista de posiciones a un array NumPy
positions = np.array(positions).astype(np.float32)

# Definir criterios para el algoritmo KMeans de OpenCV
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Número de clusters deseados
k = 10

# Aplicar KMeans a las posiciones de cámara
_, labels, centers = cv2.kmeans(positions, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Visualizar las posiciones de cámara y los centros de cluster en un gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Colores por cluster
colors = plt.cm.tab10(labels.flatten())

# Dibujar las posiciones de cámara
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=40, label="Camera Positions")

# Dibujar los centros de cluster con marcadores grandes
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=150, marker='x', label="Cluster Centers")

# Configurar título y ejes
ax.set_title('Clustered Camera Positions (K=10)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Mostrar el gráfico
plt.show()
