import os
import pandas as pd
import shutil

csv_path = 'dataset_original/HAM10000_metadata.csv'
images_path = 'dataset_original/HAM10000_images'
output_path = 'dataset_ordenado'

df = pd.read_csv(csv_path)

# Crear carpetas por clase
for clase in df['dx'].unique():
    os.makedirs(os.path.join(output_path, clase), exist_ok=True)

# Mover imágenes
for _, row in df.iterrows():
    img = row['image_id'] + '.jpg'
    src = os.path.join(images_path, img)
    dst = os.path.join(output_path, row['dx'], img)
    if os.path.exists(src):
        shutil.copy(src, dst)

print("✅ Imágenes organizadas exitosamente.")
