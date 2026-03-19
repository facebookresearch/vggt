import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ast
from tkinter import filedialog, Tk, simpledialog

def ask_cluster_number():
    """Pide al usuario el número de clusters mediante una ventana (basado en v3)"""
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

def convert_to_plot(points):
    """
    Transforma coordenadas de OpenCV/VGGT al sistema de Mapa (Z-up).
    Invierte Y y Z para que el dron apunte hacia el suelo.
    """
    if points.ndim == 1:
        return np.array([points[0], -points[1], -points[2]])
    res = points.copy().astype(float)
    res[:, 1] = -res[:, 1] 
    res[:, 2] = -res[:, 2] 
    return res

def view_cluster_vx():
    # 1. Selección de archivo y configuración de clusters
    root = Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(title="Selecciona el archivo camera_data_vx.csv")
    root.destroy()
    if not csv_path: return
    
    n_clusters = ask_cluster_number()
    if n_clusters is None: n_clusters = 5

    df = pd.read_csv(csv_path)
    
    # 2. Clustering por Altura y Scale Match (RS) según Yu et al.
    features = df[['height', 'rs_val']].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    df['cluster'] = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(scaled_features)

    # 3. Cálculo de Escala Proporcional (8% de la extensión de la escena)
    positions_raw = df[['pos_x', 'pos_y', 'pos_z']].values
    scene_span = np.max(np.ptp(positions_raw, axis=0))
    base_scale = scene_span * 0.08 if scene_span > 0 else 1.0

    # 4. Configuración del Gráfico 3D
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('tab20')

    all_visible_vertices = []

    # 5. Representación por Grupos
    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        if cluster_data.empty: continue
        color = cmap(cluster_id % 20)
        
        # A. Trayectoria: Puntos del cluster
        pts_raw = cluster_data[['pos_x', 'pos_y', 'pos_z']].values
        plot_pts = convert_to_plot(pts_raw)
        ax.scatter(plot_pts[:,0], plot_pts[:,1], plot_pts[:,2], 
                   color=color, s=20, alpha=0.2, label=f'Grupo {cluster_id}')
        all_visible_vertices.append(plot_pts)

        # B. Representante del Cluster (Centroide)
        cluster_scaled_feats = scaled_features[df['cluster'] == cluster_id]
        centroid = np.mean(cluster_scaled_feats, axis=0)
        distances = np.linalg.norm(cluster_scaled_feats - centroid, axis=1)
        rep_idx = cluster_data.index[np.argmin(distances)]
        rep_row = df.loc[rep_idx]
        
        # C. Calcular altura promedio del grupo para la etiqueta
        avg_height = cluster_data['height'].mean()

        # D. Datos de Pose
        pos = np.array([rep_row['pos_x'], rep_row['pos_y'], rep_row['pos_z']])
        R_raw = ast.literal_eval(rep_row['R_world_flat'])
        R_wc = np.array(R_raw).reshape(3, 3)
        plot_pos = convert_to_plot(pos)

        # E. Dirección de Captura (Flecha)
        dir_vec = R_wc @ np.array([0, 0, 1])
        plot_dir = np.array([dir_vec[0], -dir_vec[1], -dir_vec[2]])
        ax.quiver(plot_pos[0], plot_pos[1], plot_pos[2], 
                  plot_dir[0], plot_dir[1], plot_dir[2], 
                  length=base_scale * 2.5, color=color, linewidth=3, arrow_length_ratio=0.3)

        # F. Pirámide de Ángulo Focal (Frustum)
        s = base_scale
        w = (rep_row['principal_x'] / rep_row['focal_x']) * s if 'focal_x' in rep_row else s
        h = (rep_row['principal_y'] / rep_row['focal_y']) * s if 'focal_y' in rep_row else s * 0.75
        z_dist = s * 1.5

        local_v = np.array([
            [0, 0, 0],               
            [w, h, z_dist], [-w, h, z_dist], [-w, -h, z_dist], [w, -h, z_dist] 
        ]).T
        
        world_v = (R_wc @ local_v).T + pos
        plot_v = convert_to_plot(world_v)
        all_visible_vertices.append(plot_v)
        
        faces = [
            [plot_v[0], plot_v[1], plot_v[2]], [plot_v[0], plot_v[2], plot_v[3]],
            [plot_v[0], plot_v[3], plot_v[4]], [plot_v[0], plot_v[4], plot_v[1]],
            [plot_v[1], plot_v[2], plot_v[3], plot_v[4]]
        ]
        
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=0.5, edgecolors='k', linewidths=0.5))
        
        # G. Etiqueta con Altura Relativa
        label_text = f"G{cluster_id}: H={avg_height:.2f}m"
        ax.text(plot_pos[0], plot_pos[1], plot_pos[2] + (base_scale * 0.8), 
                label_text, fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, edgecolor=color))

    # 6. Ajuste dinámico de límites
    all_v = np.vstack(all_visible_vertices)
    mid = np.mean(all_v, axis=0)
    max_range = np.max(np.ptp(all_v, axis=0)) / 1.6 # Ajuste para incluir etiquetas
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_title(f"Dashboard de Poses ({n_clusters} Clusters) - Altura Relativa y Scale Match", fontsize=14)
    ax.set_xlabel("X (Lateral)"); ax.set_ylabel("-Y (Profundidad)"); ax.set_zlabel("Z (Altura)")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_cluster_vx()