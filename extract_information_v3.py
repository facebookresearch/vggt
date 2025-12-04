"""
Script mejorado para procesar imágenes con el modelo VGGT.
Versión: extract_information_v3

Características:
- Procesa imágenes en secuencia (batch) para consistencia global.
- Genera y guarda mapas de profundidad (Depth Maps) coloreados (Turbo).
- Extrae parámetros intrínsecos (f, cx, cy) y extrínsecos (R, T).
- Calcula la posición del centro de la cámara (C) en el mundo.
- Guarda toda la información en un CSV, incluyendo la ruta de los mapas de profundidad.

Nota: Aunque se extrae la componente Z de la posición, no se interpreta explícitamente como "altura absoluta" 
debido a la naturaleza de escala ambigua en reconstrucción monocular/sin referencia métrica.

Requisitos:
- torch, torchvision
- PIL, matplotlib, pandas, numpy
- vggt
"""

import os
import torch
import numpy as np
import pandas as pd
import PIL.Image
import matplotlib.pyplot as plt
import math
from tkinter import filedialog, Tk

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def select_folder(prompt):
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=prompt)
    root.destroy()
    return folder_path

def rotation_matrix_to_euler_angles(R):
    """
    Convierte una matriz de rotación 3x3 a ángulos de Euler (pitch, yaw, roll) en radianes.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def process_images_v3(input_folder, output_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"Cargando modelo VGGT en {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # Obtener lista de archivos
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    if not image_files:
        print("No se encontraron imágenes en la carpeta seleccionada.")
        return

    print(f"Encontradas {len(image_files)} imágenes. Cargando secuencia...")

    img_paths = [os.path.join(input_folder, f) for f in image_files]
    
    # Cargar secuencia [S, 3, H, W]
    try:
        images_seq = load_and_preprocess_images(img_paths).to(device)
    except Exception as e:
        print(f"Error cargando imágenes: {e}")
        return

    # Batch dimension [1, S, 3, H, W]
    images_batch = images_seq.unsqueeze(0)

    print("Ejecutando inferencia VGGT...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images_batch)

    # --- Procesamiento de Parámetros ---
    model_res = images_seq.shape[-2:] # (518, 518)
    extrinsic_batch, intrinsic_batch = pose_encoding_to_extri_intri(predictions["pose_enc"], model_res)
    
    # [S, 3, 4]
    extrinsics = extrinsic_batch.squeeze(0).cpu().numpy()
    intrinsics = intrinsic_batch.squeeze(0).cpu().numpy()

    # --- Procesamiento de Profundidad ---
    # [S, H, W]
    depth_batch = predictions["depth"].squeeze(0).squeeze(-1)
    
    data_records = []
    print("Procesando datos...")

    for i, img_file in enumerate(image_files):
        img_path = img_paths[i]
        
        # Dimensiones originales para escalado
        with PIL.Image.open(img_path) as img_pil:
            orig_w, orig_h = img_pil.size
        
        # 1. Guardar Depth Map
        depth_map = depth_batch[i].float().cpu().numpy()
        
        # Visualización (Inverse Depth)
        depth_map_safe = np.where(depth_map > 1e-5, depth_map, 1e-5)
        inverse_depth = 1.0 / depth_map_safe
        
        max_inv = min(inverse_depth.max(), 1 / 0.1)
        min_inv = max(1 / 250, inverse_depth.min())
        
        if max_inv > min_inv:
            inv_norm = (inverse_depth - min_inv) / (max_inv - min_inv)
        else:
            inv_norm = np.zeros_like(inverse_depth)

        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inv_norm)[..., :3] * 255).astype(np.uint8)
        
        depth_filename = f"depth_{os.path.splitext(img_file)[0]}.jpeg"
        output_depth_path = os.path.join(output_folder, depth_filename)
        PIL.Image.fromarray(color_depth).save(output_depth_path, format="JPEG", quality=90)

        # 2. Intrínsecos (Escalado)
        model_h, model_w = model_res
        intrinsic_np = intrinsics[i]
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        
        fx = intrinsic_np[0, 0] * scale_x
        fy = intrinsic_np[1, 1] * scale_y
        cx = intrinsic_np[0, 2] * scale_x
        cy = intrinsic_np[1, 2] * scale_y
        
        # 3. Extrínsecos
        extrinsic_np = extrinsics[i]
        R = extrinsic_np[:3, :3]
        t = extrinsic_np[:3, 3]
        
        # Posición de Cámara en el Mundo: C = -R^T * t
        C = -R.T @ t
        
        # Orientación (Euler)
        euler = rotation_matrix_to_euler_angles(R)
        
        record = {
            "image_name": img_file,
            "depth_map_file": depth_filename,
            # Intrínsecos
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            # Extrínsecos (Posición)
            "tx": C[0],
            "ty": C[1],
            "tz": C[2], # Componente Z de la posición
            # Extrínsecos (Orientación)
            "pitch": euler[0],
            "yaw": euler[1],
            "roll": euler[2],
            # Raw Data
            "R_flat": R.flatten().tolist(),
            "t_flat": t.flatten().tolist()
        }
        data_records.append(record)
        print(f"Procesado: {img_file}")

    # Guardar CSV
    csv_output_path = os.path.join(output_folder, "camera_parameters_v3.csv")
    df = pd.DataFrame(data_records)
    df.to_csv(csv_output_path, index=False)
    print(f"Datos guardados en: {csv_output_path}")

if __name__ == "__main__":
    print("Selecciona carpeta de imágenes...")
    input_folder = select_folder("Entrada")
    if not input_folder: exit()
        
    print("Selecciona carpeta de salida...")
    output_folder = select_folder("Salida")
    if not output_folder: exit()

    process_images_v3(input_folder, output_folder)
