import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ast
import cv2
from tkinter import filedialog, Tk, simpledialog
import sys
import os

# --- UTILIDADES ---

def select_file(prompt):
    """Abre una ventana para seleccionar el archivo CSV"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title=prompt,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def ask_cluster_number():
    """Pide al usuario el número de clusters mediante una ventana"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    num = simpledialog.askinteger(
        "Configuración de Clusters", 
        "Introduce el número de clusters para agrupar las cámaras:",
        parent=root,
        minvalue=1, 
        maxvalue=50, 
        initialvalue=5
    )
    root.destroy()
    return num

def convert_opencv_to_plot(points):
    """
    Transforma coordenadas del sistema OpenCV (cámara) al sistema Plot/Mapa (gráfico).
    
    MODO UAV/DRONE (Nadir):
    Asumimos que el sistema de coordenadas del mundo (Frame 1) tiene:
    - Z+ apuntando hacia el suelo (Vista de la cámara).
    - Y+ apuntando hacia "Abajo" de la imagen (Sur/Atrás).
    - X+ apuntando a la Derecha.
    
    Queremos transformar a un sistema de mapa (NED/ENU modificado):
    - Plot Z+ = Altura (Hacia arriba).
    - Plot Y+ = Norte/Adelante.
    - Plot X+ = Derecha.
    
    Mapeo:
    Plot_X = CV_X
    Plot_Y = -CV_Y  (CV Y es Back -> Plot Y es Forward)
    Plot_Z = -CV_Z  (CV Z es Down -> Plot Z es Up)
    """
    if points.ndim == 1:
        x, y, z = points
        return np.array([x, -y, -z])
        
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    new_points = np.stack([x, -y, -z], axis=1)
    return new_points

def plot_camera(ax, position, rotation_matrix, scale=0.5, color='blue', label=None):
    """
    Dibuja la cámara: Pirámide (Frustum) + Flecha de dirección (Vector)
    Aplica conversión de coordenadas de OpenCV (Y-down) a Plot (Z-up).
    """
    # 1. DIBUJAR FRUSTUM (PIRÁMIDE)
    w = scale
    h = scale * 0.75
    z = scale * 0.8
    
    # Vértices locales (Punta en 0,0,0)
    # Z+ es hacia adelante en el frame de la cámara
    local_verts = np.array([
        [0, 0, 0],          # Ojo
        [w, h, z], [-w, h, z], [-w, -h, z], [w, -h, z] # Base
    ]).T

    # Transformar al mundo (OpenCV): R * P_local + C
    world_verts = np.dot(rotation_matrix, local_verts) + position.reshape(3, 1)
    
    # CONVERTIR AL SISTEMA DEL GRÁFICO (Z-Up)
    plot_verts = convert_opencv_to_plot(world_verts.T).T
    plot_pos = convert_opencv_to_plot(position)

    # Caras de la pirámide
    # Los índices son sobre las columnas de plot_verts
    verts_indices = [
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], # Lados
        [1, 2, 3, 4] # Tapa trasera
    ]
    poly_3d = [[tuple(plot_verts[:, i]) for i in indices] for indices in verts_indices]

    # Agregar la pirámide semitransparente
    ax.add_collection3d(Poly3DCollection(poly_3d, facecolors=color, linewidths=0.5, edgecolors='k', alpha=0.3))
    
    # 2. DIBUJAR VECTOR DE DIRECCIÓN (FLECHA)
    # El eje Z local es [0, 0, 1].
    # Vector dirección en Mundo CV = R * [0,0,1]
    direction_cv = rotation_matrix @ np.array([0, 0, 1])
    
    # Convertir vector a Plot usando la función helper
    direction_plot = convert_opencv_to_plot(direction_cv)
    
    # Dibujar flecha (quiver)
    arrow_len = scale * 1.5 
    ax.quiver(
        plot_pos[0], plot_pos[1], plot_pos[2], # Origen
        direction_plot[0], direction_plot[1], direction_plot[2], # Vector
        length=arrow_len, color=color, linewidth=1.5, arrow_length_ratio=0.3
    )

    # 3. DIBUJAR CENTRO Y ETIQUETA
    ax.scatter(plot_pos[0], plot_pos[1], plot_pos[2], color='black', s=10)
    if label:
        ax.text(plot_pos[0], plot_pos[1], plot_pos[2], label, fontsize=8)

# --- VISUALIZACIÓN COMPLETA (V3) ---

def view_analysis_v3():
    # --- 1. Selección de Archivo ---
    print("Abriendo ventana para seleccionar archivo CSV...")
    csv_path = select_file("Selecciona el archivo 'camera_data_vx.csv'")
    
    if not csv_path:
        print("Cancelado por el usuario.")
        return

    # --- 2. Selección de Clusters ---
    num_clusters = ask_cluster_number()
    if num_clusters is None:
        print("No se introdujo número. Usando por defecto: 5")
        num_clusters = 5

    # Cargar datos
    print(f"Cargando datos de {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)
    
    # Extraer posiciones (OpenCV World Coords)
    positions = df[['pos_x', 'pos_y', 'pos_z']].values.astype(np.float32)
    
    # Extraer alturas (si existen, sino calcular desde Z)
    if 'height' in df.columns:
        heights = df['height'].values
    else:
        # Fallback para CSVs antiguos: altura = -Z (asumiendo nadir)
        heights = -positions[:, 2]

    # Parsear rotaciones
    rotations = []
    for r_str in df['R_world_flat']:
        r_list = ast.literal_eval(r_str) if isinstance(r_str, str) else r_str
        rotations.append(np.array(r_list).reshape(3, 3))

    # --- 3. Agrupamiento (Clustering) ---
    if len(positions) < num_clusters:
        num_clusters = len(positions)

    # Usamos KMeans para agrupar por POSICIÓN espacial
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(positions, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()

    # --- 4. Configuración de Gráficos (2 Subplots) ---
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"Análisis de información de imágenes extraidas por VGGT", fontsize=16)

    # Mapa de colores
    cmap = plt.get_cmap("tab20") 

    # -------------------------------------------------------------------------
    # GRÁFICO 1: VISUALIZACIÓN 3D DE CLUSTERS (REPRESENTATIVOS)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("1. Clusters 3D (Representantes)")
    
    # Escala visual
    scene_span = np.max(np.ptp(positions, axis=0))
    cam_scale = scene_span * 0.08 if scene_span > 0 else 0.1
    
    print(f"Graficando {num_clusters} grupos representativos...")
    
    # Iterar por cada cluster para dibujar SOLO el representante
    colors_per_cluster = [cmap(k / max(num_clusters, 1)) for k in range(num_clusters)]
    
    for k in range(num_clusters):
        cluster_indices = np.where(labels == k)[0]
        if len(cluster_indices) == 0: continue
            
        cluster_center = centers[k]
        dists = np.linalg.norm(positions[cluster_indices] - cluster_center, axis=1)
        closest_idx = cluster_indices[np.argmin(dists)]
        rep_rotation = rotations[closest_idx]
        
        plot_camera(ax1, cluster_center, rep_rotation, scale=cam_scale*2.0, color=colors_per_cluster[k], label=f"C{k}")

    # Etiquetas Ejes 3D
    ax1.set_xlabel('X (Lateral)')
    ax1.set_ylabel('Y (Profundidad)')
    ax1.set_zlabel('Z (Altura relativa)')
    
    # Ajuste de Aspect Ratio 3D
    limits = np.array([ax1.get_xlim3d(), ax1.get_ylim3d(), ax1.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax1.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax1.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax1.set_zlim3d([origin[2] - radius, origin[2] + radius])

    # -------------------------------------------------------------------------
    # GRÁFICO 2: ANÁLISIS DE ALTURA RELATIVA
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(122)
    ax2.set_title("2. Análisis de altura de vuelo relativa (Z)")
    
    # Estadísticas
    mean_h = np.mean(heights)
    std_h = np.std(heights)
    
    # Scatter de alturas por imagen (eje X = índice imagen)
    # Coloreamos por cluster para correlacionar
    x_indices = np.arange(len(heights))
    point_colors = [colors_per_cluster[l] for l in labels]
    
    ax2.scatter(x_indices, heights, c=point_colors, alpha=0.7, label='Imágenes')
    
    # Líneas de referencia
    ax2.axhline(mean_h, color='green', linestyle='-', linewidth=2, label=f'Promedio: {mean_h:.2f}m')
    ax2.axhline(mean_h + std_h, color='red', linestyle='--', alpha=0.5, label=f'+1 STD: {mean_h+std_h:.2f}m')
    ax2.axhline(mean_h - std_h, color='red', linestyle='--', alpha=0.5, label=f'-1 STD: {mean_h-std_h:.2f}m')
    
    # Relleno de desviación estándar
    ax2.fill_between(x_indices, mean_h - std_h, mean_h + std_h, color='gray', alpha=0.1)
    
    ax2.set_xlabel("Índice de Imagen")
    ax2.set_ylabel("Altura relativa (m)")
    ax2.legend(loc='best', fontsize='small')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Finalizar
    plt.tight_layout()
    print("Mostrando dashboard de análisis...")
    plt.show()

if __name__ == "__main__":
    view_analysis_v3()
