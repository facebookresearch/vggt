import os
import torch
import numpy as np
import pandas as pd
from tkinter import filedialog, Tk
import sys

# Importaciones de VGGT (asegúrate de ejecutar esto desde la raíz del proyecto vggt)
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except ImportError:
    print("Error: No se encuentran los módulos de VGGT. Asegúrate de ejecutar este script desde la carpeta raíz 'vggt-data-augmentation'.")
    sys.exit(1)

def select_folder(prompt):
    """Abre una ventana para seleccionar carpeta"""
    root = Tk()
    root.withdraw() # Ocultar la ventana principal de Tkinter
    root.attributes('-topmost', True) # Forzar la ventana al frente
    folder_path = filedialog.askdirectory(title=prompt)
    root.destroy()
    return folder_path

def extract_information_vx():
    # --- 1. Selección de Carpetas ---
    print("Por favor, selecciona la carpeta con las IMÁGENES de entrada...")
    image_folder = select_folder("Selecciona la carpeta con las IMÁGENES")
    
    if not image_folder:
        print("No se seleccionó carpeta de entrada. Cancelando.")
        return

    print("Por favor, selecciona la carpeta donde guardar el CSV de salida...")
    output_folder = select_folder("Selecciona la carpeta de SALIDA (para guardar el CSV)")
    
    if not output_folder:
        print("No se seleccionó carpeta de salida. Cancelando.")
        return

    output_csv = os.path.join(output_folder, "camera_data_vx.csv")

    # --- 2. Configuración del Modelo ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"Usando dispositivo: {device}")

    print("Cargando modelo VGGT...")
    try:
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    except Exception as e:
        print(f"Nota: Carga automática falló ({e}), intentando carga manual...")
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        model = model.to(device)
    model.eval()

    # --- 3. Procesamiento ---
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                          if f.lower().endswith(valid_exts)])
    
    if not image_files:
        print(f"Error: No se encontraron imágenes válidas en {image_folder}")
        return

    print(f"Procesando {len(image_files)} imágenes...")
    
    # Cargar imágenes
    images_tensor = load_and_preprocess_images(image_files).to(device)

    # Inferencia
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            if images_tensor.ndim == 4: 
                images_input = images_tensor.unsqueeze(0)
            else:
                images_input = images_tensor
            
            predictions = model(images_input)
            pose_enc = predictions["pose_enc"]

    # Descodificar poses
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])
    extrinsics = extrinsics.squeeze(0).float().cpu().numpy()
    intrinsics = intrinsics.squeeze(0).float().cpu().numpy()

    # --- 4. Guardar Datos ---
    data_records = []
    for i, img_path in enumerate(image_files):
        K = intrinsics[i]
        E = extrinsics[i] # [R|t] (Cámara <- Mundo)
        
        R = E[:3, :3]
        t = E[:3, 3]

        # Cálculo de posición real en el mundo: C = -R^T * t
        camera_center_world = -np.dot(R.T, t)
        
        # Rotación para visualización (Mundo <- Cámara)
        # Esta es la orientación de la cámara en el mundo
        R_wc = R.T
        
        # Calcular Ángulos de Euler (Yaw, Pitch, Roll) a partir de R_wc
        # Asumimos convención XYZ o similar. Para cámaras suele ser útil Pitch, Yaw, Roll.
        # Una implementación robusta de rotación a Euler (ZYX convention: Z=Yaw, Y=Pitch, X=Roll)
        import math
        sy = math.sqrt(R_wc[0, 0] * R_wc[0, 0] + R_wc[1, 0] * R_wc[1, 0])
        singular = sy < 1e-6
        if not singular:
            x_rot = math.atan2(R_wc[2, 1], R_wc[2, 2])
            y_rot = math.atan2(-R_wc[2, 0], sy)
            z_rot = math.atan2(R_wc[1, 0], R_wc[0, 0])
        else:
            x_rot = math.atan2(-R_wc[1, 2], R_wc[1, 1])
            y_rot = math.atan2(-R_wc[2, 0], sy)
            z_rot = 0

        # Convertir a grados
        roll = np.degrees(x_rot)
        pitch = np.degrees(y_rot)
        yaw = np.degrees(z_rot)

        # Altura relativa
        # Para datos de UAV/Drones (VGGT suele alinear Z con la vista), 
        # si la cámara mira hacia abajo (Z+), la altura/altitud varía en el eje Z.
        # Asumimos que moverse en -Z es subir.
        height_rel = -camera_center_world[2]

        # --- 5. Guardar Mapa de Profundidad ---
        depth_rel_path = ""
        if "depth" in predictions:
            # Obtener el mapa de profundidad para este índice
            # depth_tensor shape: B, S, H, W, 1
            # predictions["depth"][0, i, :, :, 0]
            d_map = predictions["depth"][0, i, :, :, 0].float().cpu().numpy()
            
            # Normalizar para visualización (Inverse Depth suele verse mejor)
            d_map = d_map + 1e-6
            inv_depth = 1.0 / d_map
            vmax = np.percentile(inv_depth, 95)
            vmin = np.percentile(inv_depth, 5)
            norm_depth = (inv_depth - vmin) / (vmax - vmin + 1e-8)
            norm_depth = np.clip(norm_depth, 0, 1)
            
            # Colorear con mapa de calor (Turbo es bueno para profundidad)
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("turbo")
            color_depth = (cmap(norm_depth)[..., :3] * 255).astype(np.uint8)
            
            # Guardar imagen
            from PIL import Image
            depth_filename = f"depth_{os.path.basename(img_path).split('.')[0]}.png"
            depth_save_path = os.path.join(output_folder, "depth_maps", depth_filename)
            os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)
            
            Image.fromarray(color_depth).save(depth_save_path)
            depth_rel_path = depth_filename

        data_records.append({
            "image_name": os.path.basename(img_path).split('.')[0], # Nombre sin extensión para referencia más limpia
            "full_path": img_path,
            "depth_map_path": depth_rel_path,
            "focal_x": K[0, 0],
            "focal_y": K[1, 1],
            "principal_x": K[0, 2],
            "principal_y": K[1, 2],
            "pos_x": camera_center_world[0],
            "pos_y": camera_center_world[1],
            "pos_z": camera_center_world[2],
            "height": height_rel,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "R_world_flat": R_wc.flatten().tolist()
        })

    df = pd.DataFrame(data_records)
    df.to_csv(output_csv, index=False)
    print(f"¡Éxito! Datos guardados en: {output_csv}")

if __name__ == "__main__":
    extract_information_vx()