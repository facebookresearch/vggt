"""
Script de visualización para clusters de cámaras.
Versión: view_cluster_v3

Características:
- Visualiza la posición de las cámaras en 3D.
- Muestra la orientación mediante frustums.
- NO muestra trayectoria (líneas).
- NO enfatiza altura (ejes o análisis de Z), solo geometría relativa.

Requisitos:
- pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ast
from tkinter import filedialog, Tk

def select_file(prompt):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt, filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    return file_path

def draw_camera_frustum(ax, R, C, scale=1.0, color='cyan'):
    w = 0.5 * scale
    h = 0.35 * scale
    d = 1.0 * scale 
    
    # Vértices en coordenadas de cámara (OpenCV)
    v_cam = np.array([
        [0, 0, 0],          # Centro
        [-w, -h, d],        # TL
        [w, -h, d],         # TR
        [w, h, d],          # BR
        [-w, h, d]          # BL
    ])
    
    # Transformar a mundo: P_world = R.T @ P_cam + C
    R_wc = R.T
    v_world = (R_wc @ v_cam.T).T + C
    
    verts = [
        [v_world[0], v_world[1], v_world[2]], 
        [v_world[0], v_world[2], v_world[3]], 
        [v_world[0], v_world[3], v_world[4]], 
        [v_world[0], v_world[4], v_world[1]], 
        [v_world[1], v_world[2], v_world[3], v_world[4]] 
    ]
    
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=1, edgecolors='k', alpha=0.25))
    
    # Eje Z local (dirección de vista)
    z_end = (R_wc @ np.array([0, 0, d*1.5])) + C
    ax.plot([C[0], z_end[0]], [C[1], z_end[1]], [C[2], z_end[2]], color='blue', linewidth=1)

def visualize_cluster_v3(csv_path):
    print(f"Cargando {csv_path}...")
    df = pd.read_csv(csv_path)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Usamos tx, ty, tz calculados en extract_information_v3
    coords = df[['tx', 'ty', 'tz']].values
    
    # Escala automática
    if len(coords) > 1:
        max_range = np.max(np.ptp(coords, axis=0))
        scale = max_range * 0.05 if max_range > 0 else 1.0
    else:
        scale = 1.0

    print(f"Visualizando {len(df)} cámaras...")
    
    for idx, row in df.iterrows():
        C = np.array([row['tx'], row['ty'], row['tz']])
        
        if 'R_flat' in row:
            R_flat = ast.literal_eval(row['R_flat']) if isinstance(row['R_flat'], str) else row['R_flat']
            R = np.array(R_flat).reshape(3, 3)
            
            draw_camera_frustum(ax, R, C, scale=scale, color='cyan')
            ax.text(C[0], C[1], C[2], str(idx+1), color='black', fontsize=8)

    # Configuración
    ax.set_title('Posiciones de Cámara (VGGT)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') # Mantenemos eje Z para la geometría 3D, pero sin énfasis en "Altura"
    
    # Límites cúbicos
    if len(coords) > 0:
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        if radius == 0: radius = 1.0
        
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    
    plt.show()

if __name__ == "__main__":
    print("Selecciona archivo CSV...")
    csv_path = select_file("Selecciona camera_parameters_v3.csv")
    if csv_path:
        visualize_cluster_v3(csv_path)
