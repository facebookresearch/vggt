"""
Script de visualización mejorado para clusters de cámaras y trayectorias.
Versión: view_cluster_v2

Funcionalidades:
- Carga datos de 'camera_parameters_v2.csv'.
- Visualiza posiciones de cámara en 3D.
- Representa la orientación de cada cámara mediante frustums (pirámides).
- Dibuja la trayectoria conectando las cámaras secuencialmente.
- Estilo visual similar a herramientas de fotogrametría (Matlab/COLMAP).

Requisitos:
- pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ast
import os
from tkinter import filedialog, Tk

def select_file(prompt):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt, filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    return file_path

def draw_camera_frustum(ax, R, C, scale=1.0, color='cyan'):
    """
    Dibuja un frustum de cámara en el gráfico 3D ax.
    R: Matriz de rotación World->Camera (3x3)
    C: Centro de la cámara en coordenadas World (3,)
    scale: Tamaño del frustum
    """
    # Definir vértices del frustum en el sistema de coordenadas de la cámara (OpenCV: Z forward, Y down)
    # Vértice en el origen (centro óptico)
    # Y 4 vértices en el plano de imagen (simulado)
    w = 0.5 * scale
    h = 0.35 * scale
    d = 1.0 * scale # Distancia focal simulada para visualización
    
    # Vértices en coordenadas de cámara
    # Puntos: Origen, Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # OpenCV: X right, Y down, Z forward
    v_cam = np.array([
        [0, 0, 0],          # 0: Centro
        [-w, -h, d],        # 1: TL
        [w, -h, d],         # 2: TR
        [w, h, d],          # 3: BR
        [-w, h, d]          # 4: BL
    ])
    
    # Transformar a coordenadas del mundo
    # P_world = R_cam2world @ P_cam + C
    # R es World->Camera, así que R_cam2world es R.T
    R_wc = R.T
    v_world = (R_wc @ v_cam.T).T + C
    
    # Definir las caras del frustum para Poly3DCollection
    # Caras: Base, y 4 triángulos laterales
    verts = [
        [v_world[0], v_world[1], v_world[2]], # Cara superior (triángulo)
        [v_world[0], v_world[2], v_world[3]], # Cara derecha
        [v_world[0], v_world[3], v_world[4]], # Cara inferior
        [v_world[0], v_world[4], v_world[1]], # Cara izquierda
        [v_world[1], v_world[2], v_world[3], v_world[4]] # Base (rectángulo)
    ]
    
    # Dibujar
    # Caras transparentes
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=1, edgecolors='k', alpha=0.25))
    
    # Dibujar ejes locales para orientación (opcional, como en la imagen de ejemplo)
    # Eje Z (dirección de vista)
    z_end = (R_wc @ np.array([0, 0, d*1.5])) + C
    ax.plot([C[0], z_end[0]], [C[1], z_end[1]], [C[2], z_end[2]], color='blue', linewidth=1)
    
    # Eje Y (arriba/abajo de la cámara)
    y_end = (R_wc @ np.array([0, h*1.5, 0])) + C
    ax.plot([C[0], y_end[0]], [C[1], y_end[1]], [C[2], y_end[2]], color='green', linewidth=1)

def visualize_cluster_v2(csv_path):
    print(f"Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    positions = []
    
    # Determinar escala automática para los frustums basada en la dispersión de los puntos
    coords = df[['camera_x', 'camera_y', 'camera_z']].values
    if len(coords) > 1:
        max_range = np.max(np.ptp(coords, axis=0))
        frustum_scale = max_range * 0.05 # 5% del tamaño de la escena
    else:
        frustum_scale = 1.0

    print(f"Generando visualización 3D con {len(df)} cámaras...")
    
    for idx, row in df.iterrows():
        # Posición
        C = np.array([row['camera_x'], row['camera_y'], row['camera_z']])
        positions.append(C)
        
        # Rotación
        # Si tenemos R_flat en el CSV, lo usamos. Si no, reconstruimos desde Euler (menos preciso si no guardamos R)
        if 'R_flat' in row:
            R_flat = ast.literal_eval(row['R_flat']) if isinstance(row['R_flat'], str) else row['R_flat']
            R = np.array(R_flat).reshape(3, 3)
        else:
            # Fallback a Euler si R_flat no existe (no debería pasar con extract_information_v2)
            # TODO: Implementar conversión Euler -> Matriz si es necesario
            continue
            
        # Dibujar Frustum
        draw_camera_frustum(ax, R, C, scale=frustum_scale, color='cyan')
        
        # Etiqueta (número de imagen)
        ax.text(C[0], C[1], C[2], str(idx+1), color='black', fontsize=8)

    positions = np.array(positions)
    
    # Dibujar Trayectoria
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='red', linestyle='--', linewidth=1, label='Trayectoria')
    
    # Configuración del gráfico
    ax.set_title('Reconstrucción de Trayectoria de Cámara y Poses (VGGT)')
    ax.set_xlabel('X World')
    ax.set_ylabel('Y World')
    ax.set_zlabel('Z World (Altura)')
    
    # Ajustar límites para que sea cúbico/proporcional
    # Esto es importante para que la geometría no se vea distorsionada
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    
    ax.legend()
    plt.show()

if __name__ == "__main__":
    print("Selecciona el archivo CSV generado por extract_information_v2...")
    csv_path = select_file("Selecciona el archivo camera_parameters_v2.csv")
    
    if csv_path:
        visualize_cluster_v2(csv_path)
    else:
        print("No se seleccionó ningún archivo.")
