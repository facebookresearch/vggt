import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
from tkinter import filedialog, Tk

def select_file(prompt):
    """Abre un diálogo para seleccionar un archivo CSV."""
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=prompt, filetypes=[("CSV Files", "*.csv")])

def create_camera_frustum(R, C, scale=0.5):
    """
    Crea los vértices de una pirámide (frustum) de cámara para visualización.
    
    Args:
        R (np.array): Matriz de rotación (3x3) World-to-Camera o Camera-to-World (aquí asumimos R_wc).
        C (np.array): Centro de la cámara (3,).
        scale (float): Tamaño del frustum.
    
    Returns:
        np.array: Vértices del frustum en coordenadas del mundo.
    """
    # Definir un frustum canónico en el sistema de coordenadas de la cámara (OpenCV)
    # Origen (0,0,0) y 4 esquinas en el plano de imagen (Z positivo hacia adelante)
    w = scale
    h = scale * 0.75 
    z = scale * 1.5
    
    # Vértices en coordenadas locales de cámara
    # 0: Centro óptico
    # 1-4: Esquinas del plano de imagen
    local_frustum = np.array([
        [0, 0, 0],          # Centro
        [-w, -h, z],        # Top-Left
        [w, -h, z],         # Top-Right
        [w, h, z],          # Bottom-Right
        [-w, h, z]          # Bottom-Left
    ]).T # (3, 5)

    # Transformar al mundo: P_world = R_wc * P_cam + C
    world_frustum = (R @ local_frustum).T + C
    
    return world_frustum

def plot_cameras_vx(csv_path):
    # Cargar datos
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return
    
    # Preparar figura 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    positions = []
    
    # Calcular escala de la escena para ajustar el tamaño de los frustums automáticamente
    all_tx = df['tx'].values
    all_ty = df['ty'].values
    all_tz = df['tz'].values
    
    scene_spread = np.max([
        np.max(all_tx) - np.min(all_tx),
        np.max(all_ty) - np.min(all_ty),
        np.max(all_tz) - np.min(all_tz)
    ])
    
    # Ajustar tamaño del frustum (5% del tamaño de la escena)
    frustum_scale = scene_spread * 0.05 if scene_spread > 0 else 0.1

    print(f"Renderizando {len(df)} cámaras...")

    # Colormap basado en el índice (solo para diferenciar visualmente, no implica orden temporal en el gráfico)
    colors = plt.cm.jet(np.linspace(0, 1, len(df)))

    for i, row in df.iterrows():
        # Extraer posición
        C = np.array([row['tx'], row['ty'], row['tz']])
        positions.append(C)
        
        # Extraer Rotación (string JSON a numpy)
        try:
            R_wc = np.array(json.loads(row['rotation_matrix_wc']))
        except KeyError:
            print(f"Error: La columna 'rotation_matrix_wc' no se encuentra o tiene formato incorrecto en la fila {i}.")
            continue
        
        # Crear Frustum
        verts = create_camera_frustum(R_wc, C, scale=frustum_scale)
        
        # Definir las caras del polígono para dibujar
        # Vértices: 0=Centro, 1=TL, 2=TR, 3=BR, 4=BL
        
        # Lados de la pirámide
        sides = [
            [verts[0], verts[1], verts[2]],
            [verts[0], verts[2], verts[3]],
            [verts[0], verts[3], verts[4]],
            [verts[0], verts[4], verts[1]]
        ]
        # Base de la pirámide (plano de imagen)
        base = [[verts[1], verts[2], verts[3], verts[4]]]
        
        # Dibujar Lados (translúcidos)
        ax.add_collection3d(Poly3DCollection(sides, facecolors=colors[i], linewidths=0.5, edgecolors='k', alpha=0.15))
        # Dibujar Base (más sólida para indicar la dirección)
        ax.add_collection3d(Poly3DCollection(base, facecolors=colors[i], linewidths=0.5, edgecolors='k', alpha=0.4))
        
        # Dibujar punto central (posición exacta)
        ax.scatter(C[0], C[1], C[2], color=colors[i], s=20)

    # --- Se ha eliminado la línea de trayectoria (ax.plot) ---

    # Convertir posiciones a numpy para ajustar ejes
    positions = np.array(positions)

    # Etiquetas y Estilo
    ax.set_title(f'Cluster de Cámaras VGGT ({len(df)} vistas)', fontsize=14)
    ax.set_xlabel('X (Mundo)')
    ax.set_ylabel('Y (Mundo)')
    ax.set_zlabel('Z (Mundo)')
    
    # Ajuste de ejes para que sean isométricos (evita distorsión visual)
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
    
    # Vista inicial
    ax.view_init(elev=-70, azim=-90) 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Selecciona el archivo CSV generado (vggt_camera_data.csv)...")
    csv_file = select_file("Selecciona el archivo vggt_camera_data.csv")
    
    if csv_file:
        print(f"Visualizando: {csv_file}")
        plot_cameras_vx(csv_file)
    else:
        print("Operación cancelada.")