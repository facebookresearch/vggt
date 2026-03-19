import os
import torch
import numpy as np
import pandas as pd
import math
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import sys

# Importaciones de VGGT
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except ImportError:
    print("Error: No se encuentran los módulos de VGGT. Ejecuta desde la raíz del proyecto.")
    sys.exit(1)

def select_folder(prompt):
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(title=prompt)
    root.destroy()
    return path

def extract_information_vx(batch_size=2):
    # 1. Selección de rutas
    img_dir = select_folder("Selecciona carpeta de IMÁGENES")
    out_dir = select_folder("Selecciona carpeta de SALIDA")
    if not img_dir or not out_dir: return

    depth_dir = os.path.join(out_dir, "depth_maps")
    os.makedirs(depth_dir, exist_ok=True)

    # 2. Configuración del modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando VGGT en {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)])
    
    data_records = []

    print(f"Procesando {len(image_files)} imágenes en batches de {batch_size}...")

    # 3. Procesamiento por Batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]
        images_tensor = load_and_preprocess_images(batch_files).to(device)
        
        with torch.inference_mode():
            predictions = model(images_tensor.unsqueeze(0))
            pose_enc = predictions["pose_enc"]
            depth_data = predictions["depth"].squeeze(0).float().cpu().numpy()

        # Descodificar parámetros
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])
        extrinsics = extrinsics.squeeze(0).float().cpu().numpy()
        intrinsics = intrinsics.squeeze(0).float().cpu().numpy()

        for j, img_path in enumerate(batch_files):
            # A. Guardar mapa de profundidad
            d_map = depth_data[j, :, :, 0]
            norm_d = (d_map - d_map.min()) / (d_map.max() - d_map.min() + 1e-8)
            depth_viz = (plt.cm.turbo(norm_d)[:, :, :3] * 255).astype(np.uint8)
            depth_name = f"depth_{os.path.basename(img_path)}"
            Image.fromarray(depth_viz).save(os.path.join(depth_dir, depth_name))

            # B. Extraer Pose (Mundo <- Cámara)
            R, t = extrinsics[j][:3, :3], extrinsics[j][:3, 3]
            C_world = -np.dot(R.T, t)
            height_rel = abs(C_world[2])

            data_records.append({
                "image_name": os.path.basename(img_path),
                "depth_map_path": depth_name,
                "focal_x": intrinsics[j][0, 0],
                "principal_x": intrinsics[j][0, 2],
                "principal_y": intrinsics[j][1, 2],
                "pos_x": C_world[0], 
                "pos_y": C_world[1], 
                "pos_z": C_world[2],
                "height": height_rel,
                "R_world_flat": R.T.flatten().tolist()
            })
            
        del images_tensor, predictions, pose_enc, depth_data
        torch.cuda.empty_cache()
        print(f"Progreso: {min(i + batch_size, len(image_files))}/{len(image_files)}")

    pd.DataFrame(data_records).to_csv(os.path.join(out_dir, "camera_data_vx.csv"), index=False)
    print("Extracción completada con éxito.")

if __name__ == "__main__":
    extract_information_vx(batch_size=50)