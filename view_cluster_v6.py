import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ast
from tkinter import filedialog, Tk, simpledialog
from matplotlib.lines import Line2D

def ask_cluster_number():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    num = simpledialog.askinteger(
        "Configuración de Clusters", 
        "Introduce el número de clusters para agrupar las posiciones:",
        parent=root, minvalue=1, maxvalue=50, initialvalue=5
    )
    root.destroy()
    return num

def convert_to_plot(points):
    if points.ndim == 1:
        return np.array([points[0], -points[1], -points[2]])
    res = points.copy().astype(float)
    res[:, 1] = -res[:, 1]
    res[:, 2] = -res[:, 2]
    return res

def view_cluster_vx():
    root = Tk(); root.withdraw()
    csv_path = filedialog.askopenfilename(title="Selecciona camera_data_vx.csv")
    root.destroy()
    if not csv_path: return
    
    n_clusters = ask_cluster_number()
    if n_clusters is None: n_clusters = 5

    df = pd.read_csv(csv_path)
    
    # Clustering por posición y altura
    features = df[['pos_x', 'pos_y', 'height']].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    df['cluster'] = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(scaled_features)

    positions_raw = df[['pos_x', 'pos_y', 'pos_z']].values
    scene_span = np.max(np.ptp(positions_raw, axis=0))
    base_scale = scene_span * 0.08 if scene_span > 0 else 1.0

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('tab20')

    all_v = []
    legend_elements = [] 

    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        if cluster_data.empty: continue
        color = cmap(cluster_id % 20)
        
        # --- SE ELIMINÓ AX.SCATTER PARA NO MOSTRAR PUNTOS DISPERSOS ---

        # Representante del Cluster
        cluster_scaled_feats = scaled_features[df['cluster'] == cluster_id]
        centroid = np.mean(cluster_scaled_feats, axis=0)
        distances = np.linalg.norm(cluster_scaled_feats - centroid, axis=1)
        rep_idx = cluster_data.index[np.argmin(distances)]
        rep_row = df.loc[rep_idx]

        # Pose
        pos = np.array([rep_row['pos_x'], rep_row['pos_y'], rep_row['pos_z']])
        R_wc = np.array(ast.literal_eval(rep_row['R_world_flat'])).reshape(3, 3)
        plot_pos = convert_to_plot(pos)

        # Dirección (Flecha)
        dir_vec = R_wc @ np.array([0, 0, 1])
        plot_dir = np.array([dir_vec[0], -dir_vec[1], -dir_vec[2]])
        ax.quiver(plot_pos[0], plot_pos[1], plot_pos[2], plot_dir[0], plot_dir[1], plot_dir[2], 
                  length=base_scale * 2.5, color=color, linewidth=3)

        # Pirámide de ángulo focal
        s = base_scale
        w = (rep_row['principal_x'] / rep_row['focal_x']) * s if 'focal_x' in rep_row else s
        z_dist = s * 1.5

        local_v = np.array([[0,0,0], [w,w,z_dist], [-w,w,z_dist], [-w,-w,z_dist], [w,-w,z_dist]]).T
        world_v = (R_wc @ local_v).T + pos
        plot_v = convert_to_plot(world_v)
        all_v.append(plot_v)
        all_v.append(plot_pos.reshape(1, -1))
        
        faces = [[plot_v[0], plot_v[1], plot_v[2]], [plot_v[0], plot_v[2], plot_v[3]],
                 [plot_v[0], plot_v[3], plot_v[4]], [plot_v[0], plot_v[4], plot_v[1]],
                 [plot_v[1], plot_v[2], plot_v[3], plot_v[4]]]
        
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=0.5, edgecolors='k'))
        
        ax.text(plot_pos[0], plot_pos[1], plot_pos[2] + (base_scale * 0.8), f"G{cluster_id}", 
                fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor=color))

        legend_elements.append(Line2D([0], [0], marker='s', color='w', label=f'Grupo {cluster_id}',
                                      markerfacecolor=color, markersize=10))

    # Ajuste de límites
    flat_v = np.vstack(all_v)
    mid = np.mean(flat_v, axis=0)
    max_range = np.max(np.ptp(flat_v, axis=0)) / 1.6
    
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

    ax.set_title(f"Visualización de Grupos ({n_clusters} Clusters) - Solo Representantes", fontsize=14)
    ax.set_xlabel("X (Este)"); ax.set_ylabel("-Y (Norte)"); ax.set_zlabel("Z (Altura)")
    ax.legend(handles=legend_elements, title="Grupos de Coincidencia", loc='center left', bbox_to_anchor=(1.07, 0.5))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_cluster_vx()