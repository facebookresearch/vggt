"""
Script mejorado para procesar imágenes con el modelo VGGT y extraer parámetros de cámara detallados.
Versión: extract_information_v2

Mejoras respecto a v1:
- Procesa las imágenes como una SECUENCIA (batch) para aprovechar el mecanismo de atención de VGGT y obtener consistencia en la escala y geometría (crucial para la estimación de altura).
- Extracción explícita de parámetros intrínsecos (f, cx, cy).
- Estimación de pose de cámara (Posición X, Y, Z y Orientación R).
- Cálculo de la posición de la cámara en coordenadas del mundo (C = -R^T * t).
- Escalado correcto de intrínsecos a la resolución original de la imagen.
- Salida de ángulos de Euler para la orientación.

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
    Convención XYZ.
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

def process_images_v2(input_folder, output_folder):
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
    
    # Cargar todas las imágenes como una secuencia [S, 3, H, W]
    # Esto es CRUCIAL para que VGGT entienda la geometría global y la escala relativa entre imágenes.
    try:
        images_seq = load_and_preprocess_images(img_paths).to(device)
    except Exception as e:
        print(f"Error cargando imágenes: {e}")
        print("Intenta con menos imágenes si es un error de memoria.")
        return

    # Añadir dimensión de batch [1, S, 3, H, W]
    images_batch = images_seq.unsqueeze(0)

    print("Ejecutando inferencia VGGT (esto puede tardar)...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images_batch)

    # Obtener matrices [1, S, 3, 4] y [1, S, 3, 3]
    # Usamos la resolución del modelo (518x518) para decodificar, luego escalaremos.
    model_res = images_seq.shape[-2:] # (518, 518) usualmente
    extrinsic_batch, intrinsic_batch = pose_encoding_to_extri_intri(predictions["pose_enc"], model_res)
    
    # Eliminar dimensión de batch -> [S, 3, 4]
    extrinsics = extrinsic_batch.squeeze(0).cpu().numpy()
    intrinsics = intrinsic_batch.squeeze(0).cpu().numpy()

    data_records = []
    print("Procesando resultados y extrayendo parámetros...")

    for i, img_file in enumerate(image_files):
        img_path = img_paths[i]
        
        # Cargar dimensiones originales
        with PIL.Image.open(img_path) as img_pil:
            orig_w, orig_h = img_pil.size
        
        # Matrices actuales
        extrinsic_np = extrinsics[i] # 3x4
        intrinsic_np = intrinsics[i] # 3x3
        
        # --- 1. Escalado de Intrínsecos ---
        model_h, model_w = model_res
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        
        fx = intrinsic_np[0, 0] * scale_x
        fy = intrinsic_np[1, 1] * scale_y
        cx = intrinsic_np[0, 2] * scale_x
        cy = intrinsic_np[1, 2] * scale_y
        
        # --- 2. Extrínsecos (Pose) ---
        # Extrinsic matrix is [R | t] (World -> Camera)
        R = extrinsic_np[:3, :3]
        t = extrinsic_np[:3, 3]
        
        # Calcular posición de la cámara en el mundo (Camera Center)
        # C = -R^T * t
        camera_position = -R.T @ t
        X_c, Y_c, Z_c = camera_position
        
        # Calcular Orientación (Ángulos de Euler)
        euler_angles = rotation_matrix_to_euler_angles(R)
        pitch, yaw, roll = euler_angles
        
        record = {
            "image_name": img_file,
            "focal_length_x": fx,
            "focal_length_y": fy,
            "principal_point_x": cx,
            "principal_point_y": cy,
            "camera_x": X_c,
            "camera_y": Y_c,
            "camera_z": Z_c, # Altura estimada
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "R_flat": R.flatten().tolist(),
            "t_flat": t.flatten().tolist()
        }
        data_records.append(record)
        print(f"Img: {img_file} | Altura (Z): {Z_c:.2f}")

    # Guardar CSV
    csv_output_path = os.path.join(output_folder, "camera_parameters_v2.csv")
    df = pd.DataFrame(data_records)
    df.to_csv(csv_output_path, index=False)
    print(f"Proceso completado. Datos guardados en: {csv_output_path}")

if __name__ == "__main__":
    print("Selecciona la carpeta de entrada (imágenes)...")
    input_folder = select_folder("Selecciona la carpeta con las imágenes")
    if not input_folder:
        print("No se seleccionó carpeta de entrada.")
        exit()
        
    print("Selecciona la carpeta de salida (resultados)...")
    output_folder = select_folder("Selecciona la carpeta para guardar los resultados")
    if not output_folder:
        print("No se seleccionó carpeta de salida.")
        exit()

    process_images_v2(input_folder, output_folder)
