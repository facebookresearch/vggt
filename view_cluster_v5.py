import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import cv2
from tkinter import filedialog, Tk, simpledialog
import sys

# --- CONFIGURACIÓN ---
DEFAULT_K = 100
# ---------------------

def select_file(prompt):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt, filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    return file_path

def get_k_input(total_images):
    root = Tk()
    root.withdraw()
    k = simpledialog.askinteger("Configuración", f"Total imágenes: {total_images}\nClusters a mostrar:", 
                                parent=root, minvalue=1, maxvalue=total_images, initialvalue=min(DEFAULT_K, total_images))
    root.destroy()
    return k if k is not None else min(DEFAULT_K, total_images)

def create_camera_frustum(R, C, scale=0.5):
    """
    Crea el frustum en el mundo.
    Corrección: Asumimos convención OpenCV (Z+ es forward).
    """
    w = scale
    h = scale * 0.75 
    z = scale * 1.5
    
    # Frustum en espacio local (OpenCV: Y-down, Z-forward)
    # El vértice [0,0,0] es el centro óptico
    local_frustum = np.array([
        [0, 0, 0],          # 0: Centro
        [-w, -h, z],        # 1: Top-Left (en imagen, coord negativa Y es arriba)
        [w, -h, z],         # 2: Top-Right
        [w, h, z],          # 3: Bottom-Right
        [-w, h, z]          # 4: Bottom-Left
    ]).T 

    # Transformación Rígida P_w = R_wc * P_local + C_w
    world_frustum = (R @ local_frustum).T + C
    return world_frustum

def find_representative_cameras(df, k):
    positions = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    if len(positions) <= k: return list(range(len(positions)))

    print(f"Clustering {len(positions)} cámaras a {k} representantes...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(positions, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    representative_indices = []
    for i in range(k):
        cluster_indices = np.where(labels.flatten() == i)[0]
        if len(cluster_indices) == 0: continue
        
        # Encontrar el punto real más cercano al centroide
        dists = np.linalg.norm(positions[cluster_indices] - centers[i], axis=1)
        representative_indices.append(cluster_indices[np.argmin(dists)])
        
    return representative_indices

def plot_cameras_vx(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. Clustering
    rep_indices = find_representative_cameras(df, get_k_input(len(df)))
    df_vis = df.iloc[rep_indices].reset_index(drop=True)
    
    # 2. Configuración de Escala
    all_pos = df[['tx', 'ty', 'tz']].values
    scene_span = np.linalg.norm(all_pos.max(axis=0) - all_pos.min(axis=0))
    frustum_scale = scene_span * 0.05 if scene_span > 0 else 0.1

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Matriz para corregir la visualización (OpenCV -> OpenGL-ish para plot)
    # Invierte Y y Z para que "Arriba" sea Z+ y "Abajo" sea Z- en el plot
    visual_correction = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    positions_plot = []

    print("Generando geometría...")
    colors = plt.cm.jet(np.linspace(0, 1, len(df_vis)))

    for i, row in df_vis.iterrows():
        # Datos originales
        C_raw = np.array([row['tx'], row['ty'], row['tz']])
        R_raw = np.array(json.loads(row['rotation_matrix_wc']))
        
        # Obtener geometría del frustum en coords originales
        frustum_raw = create_camera_frustum(R_raw, C_raw, scale=frustum_scale)
        
        # APLICAR CORRECCIÓN VISUAL PARA EL GRÁFICO
        # Esto rota todo el "mundo" para que sea más intuitivo de ver
        frustum_vis = frustum_raw @ visual_correction.T
        C_vis = C_raw @ visual_correction.T
        
        positions_plot.append(C_vis)
        
        verts = frustum_vis
        sides = [[verts[0], verts[1], verts[2]], [verts[0], verts[2], verts[3]], 
                 [verts[0], verts[3], verts[4]], [verts[0], verts[4], verts[1]]]
        base = [[verts[1], verts[2], verts[3], verts[4]]]
        
        # Dibujar
        ax.add_collection3d(Poly3DCollection(sides, facecolors=colors[i], linewidths=0.5, edgecolors='k', alpha=0.15))
        
        # La base indica la dirección de la mirada. La pintamos más fuerte.
        ax.add_collection3d(Poly3DCollection(base, facecolors=colors[i], linewidths=0.5, edgecolors='k', alpha=0.6))
        ax.scatter(C_vis[0], C_vis[1], C_vis[2], color=colors[i], s=15)

    # Ajuste de ejes
    pos_arr = np.array(positions_plot)
    mid = (pos_arr.max(axis=0) + pos_arr.min(axis=0)) / 2
    max_range = (pos_arr.max(axis=0) - pos_arr.min(axis=0)).max() / 2
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_title('Cluster de Cámaras (Ejes Corregidos para Visualización)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Invertido)')
    ax.set_zlabel('Z (Invertido)')
    
    # Vista isométrica inicial
    ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    f = select_file("Selecciona vggt_camera_data.csv")
    if f: plot_cameras_vx(f)