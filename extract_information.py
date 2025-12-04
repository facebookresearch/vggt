"""
Script para procesar imágenes con el modelo VGGT de Facebook AI.

Funcionalidades:
- Estima mapas de profundidad y parámetros de cámara (matrices intrínseca y extrínseca).
- Guarda la profundidad en formato JPEG (coloreada con colormap).
- Exporta los parámetros de cámara en un archivo CSV.

Requisitos:
- torch, torchvision
- PIL, matplotlib, pandas
- vggt
"""

import os
import torch
import numpy as np
import pandas as pd
import PIL.Image
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map  # (opcional)

# ---------- Selección de carpeta mediante interfaz gráfica ----------
def select_folder(prompt):
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=prompt)
    return folder_path

# ---------- Procesamiento principal de imágenes ----------
def process_images(input_folder, output_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Cargar modelo preentrenado VGGT-1B
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Filtrar imágenes válidas
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    data_records = []

    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        images = load_and_preprocess_images([img_path]).to(device)

        # Inferencia con autocasting para mejorar rendimiento
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        # Obtener matrices de cámara
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Mover todo a CPU para guardar
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        # Procesar el mapa de profundidad
        depth = predictions["depth"].squeeze()
        inverse_depth = 1 / depth

        # Normalización para visualización
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)

        # Aplicar colormap y guardar como imagen
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
        output_depth_path = os.path.join(output_folder, f"{img_file}.jpeg")
        PIL.Image.fromarray(color_depth).save(output_depth_path, format="JPEG", quality=90)

        # Guardar parámetros de cámara a CSV
        extrinsic_flat = extrinsic.cpu().flatten().numpy().tolist()
        intrinsic_flat = intrinsic.cpu().flatten().numpy().tolist()

        data_records.append({
            "image_name": img_file,
            "extrinsic": extrinsic_flat,
            "intrinsic": intrinsic_flat
        })

        print(f"Procesado: {img_file} -> {output_depth_path}")

    # Guardar CSV con parámetros
    csv_output_path = os.path.join(output_folder, "data.csv")
    df = pd.DataFrame(data_records)
    df.to_csv(csv_output_path, index=False)
    print(f"Datos guardados en: {csv_output_path}")

# ---------- Ejecución principal ----------
if __name__ == "__main__":
    input_folder = select_folder("Selecciona la carpeta con las imágenes")
    output_folder = select_folder("Selecciona la carpeta para guardar los resultados")

    if input_folder and output_folder:
        process_images(input_folder, output_folder)
    else:
        print("No se seleccionaron carpetas válidas.")
