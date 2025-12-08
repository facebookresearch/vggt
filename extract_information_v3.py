import os
import torch
import numpy as np
import pandas as pd
from tkinter import filedialog, Tk
import json
import matplotlib.pyplot as plt
from PIL import Image
import gc  # Garbage Collector para liberar memoria

# Importaciones específicas de VGGT
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# --- CONFIGURACIÓN ---
# Reduce este número si sigues teniendo errores de memoria (ej. 10, 5, 2)
BATCH_SIZE = 15  
# ---------------------

def select_folder(prompt):
    """Abre un diálogo para seleccionar una carpeta."""
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title=prompt)

def extract_camera_parameters(extrinsic, intrinsic):
    """
    Descompone las matrices de VGGT en parámetros interpretables.
    """
    # 1. Intrínsecos
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    focal_length = (fx + fy) / 2.0

    # 2. Extrínsecos
    # La matriz E (3x4) es [R_cw | t_cw]
    R_cw = extrinsic[:3, :3]
    t_cw = extrinsic[:3, 3]

    # Posición de la cámara en el mundo: C = -R_cw^T * t_cw
    R_wc = R_cw.T
    camera_center = -np.dot(R_wc, t_cw)

    return {
        "focal_length": float(focal_length),
        "principal_point": [float(cx), float(cy)],
        "intrinsic_matrix": intrinsic.tolist(), 
        "camera_position": camera_center.tolist(),
        "rotation_matrix_wc": R_wc.tolist(),
        "rotation_matrix_cw": R_cw.tolist()
    }

def save_depth_map(depth_tensor, output_path):
    """Procesa y guarda el tensor de profundidad como una imagen coloreada."""
    depth = depth_tensor + 1e-6
    inverse_depth = 1.0 / depth

    vmax = np.percentile(inverse_depth, 95)
    vmin = np.percentile(inverse_depth, 5)
    inverse_depth_normalized = (inverse_depth - vmin) / (vmax - vmin + 1e-8)
    inverse_depth_normalized = np.clip(inverse_depth_normalized, 0, 1)

    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)

    Image.fromarray(color_depth).save(output_path, format="JPEG", quality=95)

def process_batch(model, batch_files, input_folder, output_folder, depth_out_dir, device, dtype):
    """Procesa un subconjunto de imágenes y devuelve sus registros."""
    image_paths = [os.path.join(input_folder, f) for f in batch_files]
    
    # Preprocesamiento
    try:
        images_tensor = load_and_preprocess_images(image_paths).to(device)
        if images_tensor.ndim == 4:
            images_tensor = images_tensor.unsqueeze(0) # [1, S, 3, H, W]
    except Exception as e:
        print(f"  -> Error cargando batch: {e}")
        return []

    # Inferencia
    with torch.no_grad():
        # Corrección de advertencia: usar torch.amp.autocast en lugar de torch.cuda.amp.autocast
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(images_tensor)

    # Procesar resultados del batch
    pose_enc = predictions["pose_enc"]
    img_size_hw = images_tensor.shape[-2:]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, img_size_hw)
    
    # Mover a CPU
    extrinsics = extrinsics.squeeze(0).cpu().numpy()
    intrinsics = intrinsics.squeeze(0).cpu().numpy()

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
        
    # Limpieza explícita de memoria CUDA
    del images_tensor, predictions, pose_enc, extrinsics, intrinsics
    if depths_np is not None: del depths_tensor
    torch.cuda.empty_cache()
    
    return batch_records

def process_images_vx(input_folder, output_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Usando dispositivo: {device}, Precisión: {dtype}")

    print("Cargando VGGT-1B...")
    try:
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        model.eval()
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    try:
        image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    except FileNotFoundError:
        print(f"La carpeta de entrada no existe.")
        return

    total_images = len(image_files)
    if total_images == 0:
        print("No se encontraron imágenes válidas.")
        return

    print(f"Procesando {total_images} imágenes en lotes de {BATCH_SIZE}...")
    
    depth_out_dir = os.path.join(output_folder, "depth_maps")
    os.makedirs(depth_out_dir, exist_ok=True)
    
    all_records = []

    # Bucle de procesamiento por lotes
    for i in range(0, total_images, BATCH_SIZE):
        batch_files = image_files[i : i + BATCH_SIZE]
        print(f"Procesando lote {i // BATCH_SIZE + 1}/{(total_images + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch_files)} imágenes)...")
        
        records = process_batch(model, batch_files, input_folder, output_folder, depth_out_dir, device, dtype)
        all_records.extend(records)
        
        # Forzar recolección de basura de Python
        gc.collect()

    # Exportar a CSV final
    try:
        df = pd.DataFrame(all_records)
        csv_path = os.path.join(output_folder, "vggt_camera_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nProceso completado.")
        print(f"Total procesado: {len(all_records)}/{total_images}")
        print(f"CSV guardado en: {csv_path}")
    except Exception as e:
        print(f"Error guardando el CSV final: {e}")

if __name__ == "__main__":
    print("Selecciona la carpeta de imágenes de entrada...")
    in_dir = select_folder("Seleccionar carpeta de imágenes de entrada")
    
    if not in_dir:
        print("Operación cancelada.")
    else:
        print(f"Entrada: {in_dir}")
        print("Selecciona la carpeta de salida...")
        out_dir = select_folder("Seleccionar carpeta de salida")
        
        if out_dir:
            print(f"Salida: {out_dir}")
            process_images_vx(in_dir, out_dir)
        else:
            print("Operación cancelada.")