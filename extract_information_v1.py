import os
import torch
import numpy as np
import pandas as pd
from tkinter import filedialog, Tk
import json
import matplotlib.pyplot as plt
from PIL import Image
import gc

# Importaciones específicas de VGGT
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# --- CONFIGURACIÓN ---
BATCH_SIZE = 10  # Ajustado para evitar OOM
# ---------------------

def select_folder(prompt):
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title=prompt)

def extract_camera_parameters(extrinsic, intrinsic):
    """
    Descompone las matrices de VGGT en parámetros interpretables.
    Convención: Extrínseca [R|t] transforma de Mundo -> Cámara.
    """
    # 1. Intrínsecos
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    focal_length = (fx + fy) / 2.0

    # 2. Extrínsecos
    R_cw = extrinsic[:3, :3] # Rotación Mundo -> Cámara
    t_cw = extrinsic[:3, 3]  # Translación Mundo -> Cámara

    # Calcular Pose (Cámara -> Mundo)
    # Posición de la cámara C = -R_cw^T * t_cw
    R_wc = R_cw.T
    camera_center = -np.dot(R_wc, t_cw)

    return {
        "focal_length": float(focal_length),
        "principal_point": [float(cx), float(cy)],
        "intrinsic_matrix": intrinsic.tolist(), 
        "camera_position": camera_center.tolist(),
        "rotation_matrix_wc": R_wc.tolist() # Guardamos la rotación de la pose
    }

def save_depth_map(depth_tensor, output_path):
    depth = depth_tensor + 1e-6
    inverse_depth = 1.0 / depth
    vmax = np.percentile(inverse_depth, 95)
    vmin = np.percentile(inverse_depth, 5)
    inverse_depth_normalized = (inverse_depth - vmin) / (vmax - vmin + 1e-8)
    inverse_depth_normalized = np.clip(inverse_depth_normalized, 0, 1)
    
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(color_depth).save(output_path, format="JPEG", quality=85)

def process_batch(model, batch_files, input_folder, output_folder, depth_out_dir, device, dtype):
    image_paths = [os.path.join(input_folder, f) for f in batch_files]
    
    try:
        images_tensor = load_and_preprocess_images(image_paths).to(device)
        if images_tensor.ndim == 4:
            images_tensor = images_tensor.unsqueeze(0)
    except Exception as e:
        print(f"Error cargando batch: {e}")
        return []

    with torch.no_grad():
        # Usando la sintaxis moderna de autocast para evitar warnings
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(images_tensor)

    pose_enc = predictions["pose_enc"]
    img_size_hw = images_tensor.shape[-2:]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, img_size_hw)
    
    extrinsics = extrinsics.squeeze(0).cpu().numpy().astype(np.float64) # Mayor precisión
    intrinsics = intrinsics.squeeze(0).cpu().numpy().astype(np.float64)

    depths_np = None
    if "depth" in predictions:
        depths_tensor = predictions["depth"]
        depths_np = depths_tensor.squeeze(0).squeeze(-1).cpu().numpy()

    batch_records = []
    
    for i, img_name in enumerate(batch_files):
        params = extract_camera_parameters(extrinsics[i], intrinsics[i])
        
        depth_filename = ""
        if depths_np is not None:
            depth_filename = f"depth_{os.path.splitext(img_name)[0]}.jpeg"
            depth_path = os.path.join(depth_out_dir, depth_filename)
            save_depth_map(depths_np[i], depth_path)

        record = {
            "image_name": img_name,
            "depth_map_file": depth_filename,
            "f": params["focal_length"],
            "cx": params["principal_point"][0],
            "cy": params["principal_point"][1],
            "tx": params["camera_position"][0],
            "ty": params["camera_position"][1],
            "tz": params["camera_position"][2],
            "intrinsic_matrix": json.dumps(params["intrinsic_matrix"]),
            "rotation_matrix_wc": json.dumps(params["rotation_matrix_wc"]) 
        }
        batch_records.append(record)
        
    del images_tensor, predictions, pose_enc, extrinsics, intrinsics
    if depths_np is not None: del depths_tensor # Corrección de variable
    torch.cuda.empty_cache()
    
    return batch_records

def process_images_vx(input_folder, output_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Cargando modelo en {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    total_images = len(image_files)
    print(f"Encontradas {total_images} imágenes.")

    depth_out_dir = os.path.join(output_folder, "depth_maps")
    os.makedirs(depth_out_dir, exist_ok=True)
    
    all_records = []

    for i in range(0, total_images, BATCH_SIZE):
        batch_files = image_files[i : i + BATCH_SIZE]
        print(f"Procesando {i}/{total_images}...")
        records = process_batch(model, batch_files, input_folder, output_folder, depth_out_dir, device, dtype)
        all_records.extend(records)
        gc.collect()

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_folder, "vggt_camera_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Hecho. CSV en: {csv_path}")

if __name__ == "__main__":
    print("Selecciona carpeta de entrada...")
    in_dir = select_folder("Entrada")
    if in_dir:
        print("Selecciona carpeta de salida...")
        out_dir = select_folder("Salida")
        if out_dir:
            process_images_vx(in_dir, out_dir)