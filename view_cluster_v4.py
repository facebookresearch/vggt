import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import cv2  # Usamos OpenCV para K-Means
from tkinter import filedialog, Tk, simpledialog

# --- CONFIGURACIÓN POR DEFECTO ---
DEFAULT_K = 50  # Número de clusters (cámaras a visualizar) por defecto
# ---------------------------------

def select_file(prompt):
    """Abre un diálogo para seleccionar un archivo CSV."""
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=prompt, filetypes=[("CSV Files", "*.csv")])

def get_k_input():
    """Pide al usuario el número de clusters."""
    root = Tk()
    root.withdraw()
    k = simpledialog.askinteger("Configuración", "Número de clusters (cámaras a visualizar):", 
                                 parent=root, minvalue=1, maxvalue=10000, initialvalue=DEFAULT_K)
    return k if k is not None else DEFAULT_K

def create_camera_frustum(R, C, scale=0.5):
    """Crea los vértices de una pirámide (frustum) de cámara."""
    w = scale
    h = scale * 0.75 
    z = scale * 1.5
    
    local_frustum = np.array([
        [0, 0, 0],          # Centro
        [-w, -h, z],        # Top-Left
        [w, -h, z],         # Top-Right
        [w, h, z],          # Bottom-Right
        [-w, h, z]          # Bottom-Left
    ]).T 

    # Transformar al mundo: P_world = R_wc * P_cam + C
    world_frustum = (R @ local_frustum).T + C
    return world_frustum

def find_representative_cameras(df, k):
    """
    Agrupa las cámaras por posición usando K-Means y devuelve los índices
    de las cámaras reales más cercanas a los centros de los clusters.
    """
    # Extraer posiciones como float32 para OpenCV
    positions = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    
    # Si hay menos puntos que K, devolver todos
    if len(positions) <= k:
        return list(range(len(positions)))

    print(f"Agrupando {len(positions)} cámaras en {k} clusters...")

    # Criterios de parada para K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Ejecutar K-Means
    # labels: índice del cluster para cada punto
    # centers: coordenadas del centro de cada cluster
    _, labels, centers = cv2.kmeans(positions, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    representative_indices = []
    
    # Para cada cluster, encontrar el punto original más cercano a su centro
    for i in range(k):
        # Índices de las cámaras que pertenecen al cluster 'i'
        cluster_indices = np.where(labels.flatten() == i)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        # Puntos del cluster
        cluster_points = positions[cluster_indices]
        center = centers[i]
        
        # Calcular distancias al centro
        distances = np.linalg.norm(cluster_points - center, axis=1)
        
        # Índice del punto más cercano dentro del subgrupo
        closest_local_idx = np.argmin(distances)
        
        # Índice original en el DataFrame
        closest_global_idx = cluster_indices[closest_local_idx]
        representative_indices.append(closest_global_idx)
        
    return representative_indices

def plot_cameras_vx(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return
    
    # Obtener número de clusters del usuario
    k = get_k_input()
    
    # Filtrar el DataFrame para obtener solo los representantes
    rep_indices = find_representative_cameras(df, k)
    df_vis = df.iloc[rep_indices].reset_index(drop=True)
    
    # Preparar figura
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    positions = []
    
    # Calcular escala global basada en TODAS las cámaras (para el tamaño del frustum)
    all_tx = df['tx'].values
    all_ty = df['ty'].values
    all_tz = df['tz'].values
    scene_spread = np.max([
        np.max(all_tx) - np.min(all_tx),
        np.max(all_ty) - np.min(all_ty),
        np.max(all_tz) - np.min(all_tz)
    ])
    frustum_scale = scene_spread * 0.05 if scene_spread > 0 else 0.1

    print(f"Renderizando {len(df_vis)} cámaras representativas...")

    # Colormap
    colors = plt.cm.jet(np.linspace(0, 1, len(df_vis)))

    for i, row in df_vis.iterrows():
        C = np.array([row['tx'], row['ty'], row['tz']])
        positions.append(C)
        
        try:
            R_wc = np.array(json.loads(row['rotation_matrix_wc']))
        except KeyError:
            continue
        
        verts = create_camera_frustum(R_wc, C, scale=frustum_scale)
        
        sides = [
            [verts[0], verts[1], verts[2]],
            [verts[0], verts[2], verts[3]],
            [verts[0], verts[3], verts[4]],
            [verts[0], verts[4], verts[1]]
        ]
        base = [[verts[1], verts[2], verts[3], verts[4]]]
        
        ax.add_collection3d(Poly3DCollection(sides, facecolors=colors[i], linewidths=0.5, edgecolors='k', alpha=0.25))
        ax.add_collection3d(Poly3DCollection(base, facecolors=colors[i], linewidths=0.5, edgecolors='k', alpha=0.6))
        ax.scatter(C[0], C[1], C[2], color=colors[i], s=25)

    positions = np.array(positions)

    ax.set_title(f'Cluster de Cámaras VGGT (K={k} Representantes)', fontsize=14)
    ax.set_xlabel('X (Mundo)')
    ax.set_ylabel('Y (Mundo)')
    ax.set_zlabel('Z (Mundo)')
    
    if len(positions) > 0:
        max_range = np.array([
            positions[:,0].max()-positions[:,0].min(), 
            positions[:,1].max()-positions[:,1].min(), 
            positions[:,2].max()-positions[:,2].min()
        ]).max() / 2.0

        mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
        mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
        mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=-70, azim=-90) 
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Selecciona el archivo CSV...")
    csv_file = select_file("Selecciona el archivo vggt_camera_data.csv")
    
    if csv_file:
        plot_cameras_vx(csv_file)
    else:
        print("Operación cancelada.")