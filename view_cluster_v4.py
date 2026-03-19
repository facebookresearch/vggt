import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ast
from tkinter import filedialog, Tk

def view_cluster_vx():
    # 1. Cargar datos mediante interfaz
    root = Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(title="Selecciona el archivo camera_data_vx.csv")
    root.destroy()
    if not csv_path: 
        print("No se seleccionó ningún archivo.")
        return
    
    df = pd.read_csv(csv_path)
    
    # 2. Clustering por Escala Relativa (RS) y Altura
    # Agrupamos por los parámetros de Scale Match para identificar perspectivas similares [cite: 2]
    features = df[['height', 'rs_val']].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)

    # 3. Calcular escala PROPORCIONAL de la escena
    positions = df[['pos_x', 'pos_y', 'pos_z']].values
    scene_span = np.max(np.ptp(positions, axis=0))
    
    # Ajuste de escala: 8% de la extensión total para que sea "proporcional"
    # Si la escena es muy pequeña, usamos un mínimo de 0.5 unidades
    base_scale = scene_span * 0.08 if scene_span > 0 else 0.5

    # 4. Configuración de gráfico 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('tab10')

    # 5. Representación por Clusters
    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        color = cmap(cluster_id % 10)
        
        # Dibujamos todas las cámaras del cluster como puntos pequeños (contexto)
        ax.scatter(cluster_data['pos_x'], cluster_data['pos_y'], cluster_data['pos_z'], 
                   color=color, s=15, alpha=0.3)

        # 6. Identificar el Representante del Cluster
        # Es la cámara más cercana al centroide del grupo
        centroid = kmeans.cluster_centers_[cluster_id]
        cluster_indices = df.index[df['cluster'] == cluster_id]
        cluster_scaled_feats = scaled_features[df['cluster'] == cluster_id]
        
        distances = np.linalg.norm(cluster_scaled_feats - centroid, axis=1)
        rep_idx = cluster_indices[np.argmin(distances)]
        rep_row = df.loc[rep_idx]

        # Datos de pose (Mundo <- Cámara)
        pos = np.array([rep_row['pos_x'], rep_row['pos_y'], rep_row['pos_z']])
        R_wc = np.array(ast.literal_eval(rep_row['R_world_flat'])).reshape(3, 3)

        # A. Vector de dirección (Flecha de mirada del Dron)
        # El eje Z de la cámara en el mundo indica hacia dónde apunta el lente 
        direction = R_wc @ np.array([0, 0, 1]) 
        ax.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2], 
                  length=base_scale * 2.5, color=color, arrow_length_ratio=0.3, linewidth=3)

        # B. Dibujar Frustum Proporcional (Pirámide)
        s = base_scale
        # Definimos vértices: Punta en la cámara, base proyectada hacia adelante
        local_v = np.array([
            [0, 0, 0],               # Ojo de la cámara
            [s, s, s * 1.5],         # Esquina base inf-der
            [-s, s, s * 1.5],        # Esquina base inf-izq
            [-s, -s, s * 1.5],       # Esquina base sup-izq
            [s, -s, s * 1.5]         # Esquina base sup-der
        ]).T
        
        # Rotar y trasladar al mundo
        world_v = (R_wc @ local_v).T + pos
        
        # Construir las caras de la pirámide
        faces = [
            [world_v[0], world_v[1], world_v[2]],
            [world_v[0], world_v[2], world_v[3]],
            [world_v[0], world_v[3], world_v[4]],
            [world_v[0], world_v[4], world_v[1]],
            [world_v[1], world_v[2], world_v[3], world_v[4]] # Tapa de la base
        ]
        
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=0.5, edgecolors='k', linewidths=1.2))
        
        # Etiqueta indicativa
        ax.text(pos[0], pos[1], pos[2] + (base_scale * 0.5), f"Grupo {cluster_id}", 
                fontsize=10, fontweight='bold', color='black')

    # Configuración final de ejes
    ax.set_title("Trayectoria y Orientación de Captura (VGGT)\nPirámides proporcionales por cluster", fontsize=14)
    ax.set_xlabel("X (Este)"); ax.set_ylabel("Y (Norte)"); ax.set_zlabel("Z (Altura)")
    
    # Mantener el aspecto proporcional de los ejes (Bounding Box cuadrado)
    mid_x, mid_y, mid_z = np.mean(positions, axis=0)
    max_range = scene_span / 1.8
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_cluster_vx()