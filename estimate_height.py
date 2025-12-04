"""
Script para estimar y analizar la altura de captura de las imágenes.
Utiliza los datos extraídos (parámetro Z de la posición de la cámara) en los scripts anteriores.

Funcionalidades:
- Carga el CSV de parámetros de cámara (v2 o v3).
- Calcula estadísticas de altura (promedio, mínima, máxima).
- Genera un gráfico de perfil de vuelo (alturas por imagen).
- Exporta un reporte visual.

Requisitos:
- pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import os

def select_file(prompt):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt, filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    return file_path

def analyze_heights(csv_path):
    print(f"Leyendo datos de: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return

    if 'camera_z' not in df.columns:
        print("Error: El archivo CSV no contiene la columna 'camera_z'.")
        print("Asegúrate de usar el archivo generado por extract_information_v2.py o extract_information_v3.py.")
        return

    # Obtener alturas (Asumimos que Z es la altura en el sistema de coordenadas del mundo generado)
    heights = df['camera_z']
    names = df['image_name']
    
    # Estadísticas básicas
    mean_h = np.mean(heights)
    median_h = np.median(heights)
    min_h = np.min(heights)
    max_h = np.max(heights)
    std_h = np.std(heights)

    print("\n" + "="*50)
    print("   REPORTE DE ESTIMACIÓN DE ALTURA DE CAPTURA")
    print("="*50)
    print(f"Archivo analizado: {os.path.basename(csv_path)}")
    print(f"Total de imágenes: {len(heights)}")
    print("-" * 50)
    print(f"Altura Promedio (Z_avg): {mean_h:.4f}")
    print(f"Altura Mediana:          {median_h:.4f}")
    print(f"Altura Mínima:           {min_h:.4f}")
    print(f"Altura Máxima:           {max_h:.4f}")
    print(f"Desviación Estándar:     {std_h:.4f}")
    print("="*50)

    # Mostrar alturas individuales (primeras y últimas si son muchas)
    print("\nDetalle por imagen:")
    if len(df) > 20:
        print(df[['image_name', 'camera_z']].head(10).to_string(index=False))
        print("...")
        print(df[['image_name', 'camera_z']].tail(10).to_string(index=False))
    else:
        print(df[['image_name', 'camera_z']].to_string(index=False))

    # --- Visualización ---
    plt.figure(figsize=(12, 6))
    
    # Gráfico de línea
    plt.plot(range(len(heights)), heights, marker='o', linestyle='-', color='#1f77b4', label='Altura de Cámara (Z)', linewidth=1.5, markersize=4)
    
    # Línea de promedio
    plt.axhline(y=mean_h, color='#d62728', linestyle='--', label=f'Promedio ({mean_h:.2f})', linewidth=1.5)
    
    # Relleno entre min y max para dar contexto
    plt.fill_between(range(len(heights)), min_h, max_h, color='#1f77b4', alpha=0.1)

    plt.title('Perfil de Altura de Captura (Estimación Z)', fontsize=14)
    plt.xlabel('Secuencia de Imágenes', fontsize=12)
    plt.ylabel('Altura Estimada (Unidades de Mundo)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Ajustar eje X si no son demasiadas imágenes
    if len(heights) < 30:
        plt.xticks(range(len(heights)), names, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_dir = os.path.dirname(csv_path)
    plot_path = os.path.join(output_dir, "height_estimation_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nGráfico de alturas guardado en: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Selecciona el archivo CSV de parámetros de cámara (v2 o v3)...")
    csv_path = select_file("Selecciona el archivo CSV")
    
    if csv_path:
        analyze_heights(csv_path)
    else:
        print("No se seleccionó ningún archivo.")
