import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import cv2
from tkinter import filedialog, Tk, simpledialog

# --- CONFIGURACIÓN ---
DEFAULT_K = 100
# ---------------------

def select_file(prompt):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt, filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    return file_path

def get_k_input(total):
    root = Tk()
    root.withdraw()
    val = min(DEFAULT_K, total)
    k = simpledialog.askinteger("Configuración", f"Total imágenes: {total}\nClusters a mostrar:", 
                                parent=root, minvalue=1, maxvalue=total, initialvalue=val)
    root.destroy()
    return k if k else val

def convert_opencv_to_plot(points):
    """
    Transforma coordenadas del sistema OpenCV (cámara) al sistema Plot/Mapa (gráfico).
    OpenCV: X=Right, Y=Down, Z=Forward
    Plot:   X=Right, Y=Forward, Z=Up
    
    Mapeo:
    Plot_X = CV_X
    Plot_Y = CV_Z  (Profundidad se vuelve 'Norte/Adelante' en el mapa)
    Plot_Z = -CV_Y (Abajo se vuelve negativo de 'Arriba')
    """
    # Si entra un solo punto (3,) convertir a (1,3)
    if points.ndim == 1:
        points = points.reshape(1, -1)
        
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    new_points = np.stack([x, z, -y], axis=1)
    return new_points

def create_camera_frustum(R, C, scale=0.5):
    """
    1. Genera el frustum en coordenadas locales de cámara (OpenCV).
    2. Transforma al mundo (OpenCV).
    3. (La conversión a Plot se hace después).
    """
    w = scale
    h = scale * 0.75 
    z = scale * 1.5 # Profundidad del cono
    
    # Frustum local (0=Centro óptico)
    # Z+ es hacia adelante en OpenCV
    local_frustum = np.array([
        [0, 0, 0],          
        [-w, -h, z], [w, -h, z],         
        [w, h, z], [-w, h, z]          
    ]).T 

    # Transformar al mundo usando R y C extraídos del CSV
    # P_world = R_wc * P_local + C
    world_frustum_cv = (R @ local_frustum).T + C
    return world_frustum_cv

def find_representative_cameras(df, k):
    """Agrupa cámaras cercanas para limpiar la visualización."""
    positions = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    if len(positions) <= k: return df.index.tolist()

    print(f"Agrupando {len(positions)} cámaras en {k} representantes...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(positions, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    rep_indices = []
    for i in range(k):
        idx = np.where(labels.flatten() == i)[0]
        if len(idx) == 0: continue
        # Elegir cámara real más cercana al centro del cluster
        dists = np.linalg.norm(positions[idx] - centers[i], axis=1)
        rep_indices.append(idx[np.argmin(dists)])
    return rep_indices

def plot_cameras_vx(csv_path):
    print(f"Cargando: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        return

    # Clustering
    indices = find_representative_cameras(df, get_k_input(len(df)))
    df_vis = df.iloc[indices].reset_index(drop=True)
    
    # Calcular escala de la escena para tamaño de conos
    all_pos_cv = df[['tx', 'ty', 'tz']].values
    scene_span = np.linalg.norm(all_pos_cv.max(0) - all_pos_cv.min(0))
    scale = scene_span * 0.05 if scene_span > 0 else 0.1

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    positions_plot = []
    colors = plt.cm.jet(np.linspace(0, 1, len(df_vis)))

    print("Generando visualización (Transformando ejes CV -> Plot)...")
    
    for i, row in df_vis.iterrows():
        # 1. Recuperar datos (Sistema OpenCV)
        C_cv = np.array([row['tx'], row['ty'], row['tz']])
        try: 
            R_cv = np.array(json.loads(row['rotation_matrix_wc']))
        except: continue
        
        # 2. Generar Geometría del Frustum (Sistema OpenCV)
        frustum_cv = create_camera_frustum(R_cv, C_cv, scale=scale)
        
        # 3. CONVERSIÓN DE COORDENADAS (Crucial para corregir orientación)
        # Transformamos tanto el centro como los vértices del frustum
        C_plot = convert_opencv_to_plot(C_cv).flatten()
        frustum_plot = convert_opencv_to_plot(frustum_cv)
        
        positions_plot.append(C_plot)
        
        # 4. Dibujar
        verts = frustum_plot
        # Caras laterales
        sides = [[verts[0], verts[1], verts[2]], [verts[0], verts[2], verts[3]], 
                 [verts[0], verts[3], verts[4]], [verts[0], verts[4], verts[1]]]
        # Base (Plano de imagen)
        base = [[verts[1], verts[2], verts[3], verts[4]]]
        
        ax.add_collection3d(Poly3DCollection(sides, facecolors=colors[i], alpha=0.15, edgecolors='k', linewidths=0.3))
        # La base más oscura indica hacia dónde mira la cámara
        ax.add_collection3d(Poly3DCollection(base, facecolors=colors[i], alpha=0.5, edgecolors='k', linewidths=0.5))
        ax.scatter(C_plot[0], C_plot[1], C_plot[2], color=colors[i], s=15)

    # Ajuste de Ejes
    p = np.array(positions_plot)
    if len(p) > 0:
        mid = (p.max(0) + p.min(0)) / 2
        rng = (p.max(0) - p.min(0)).max() / 2
        ax.set_xlim(mid[0]-rng, mid[0]+rng)
        ax.set_ylim(mid[1]-rng, mid[1]+rng)
        ax.set_zlim(mid[2]-rng, mid[2]+rng)

    ax.set_title('Visualización de Poses (Coordenadas Mapa: Z=Arriba)')
    ax.set_xlabel('X (Lateral)')
    ax.set_ylabel('Y (Adelante/Norte)')
    ax.set_zlabel('Z (Altura)')
    
    # Vista inicial
    ax.view_init(elev=30, azim=-45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    f = select_file("Selecciona vggt_camera_data.csv")
    if f: plot_cameras_vx(f)