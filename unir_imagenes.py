import os
import shutil

# Nombres de las carpetas
parte1 = 'HAM10000_images_part_1'
parte2 = 'HAM10000_images_part_2'
destino = 'HAM10000_images'

# Crea la carpeta destino si no existe
os.makedirs(destino, exist_ok=True)

# Mueve las imágenes de ambas carpetas a la nueva
for carpeta in [parte1, parte2]:
    for archivo in os.listdir(carpeta):
        origen = os.path.join(carpeta, archivo)
        destino_final = os.path.join(destino, archivo)

        # Evitar duplicados
        if not os.path.exists(destino_final):
            shutil.move(origen, destino_final)

print("✅ Todas las imágenes han sido movidas a 'HAM10000_images'")
