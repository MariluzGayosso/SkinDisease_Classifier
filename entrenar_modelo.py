from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import json

# ========================
# 📌 Parámetros
# ========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
DATASET_PATH = 'dataset_ordenado'
MODELO_PATH = 'modelos/Model_dermatologic.keras'
CLASES_PATH = 'modelos/clases2.json'

# ========================
# 📁 Validar ruta del dataset
# ========================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ No se encontró la carpeta '{DATASET_PATH}'")

# ========================
# 📊 Generadores de datos con aumento
# ========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ========================
# 🧠 Crear modelo con MobileNetV2
# ========================
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False  # congelar capas convolucionales

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

# ========================
# ⚙️ Compilar y entrenar modelo (etapa 1)
# ========================
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print("🔁 Entrenando modelo (fase 1, capas congeladas)...")
history = model.fit(train_gen, epochs=NUM_EPOCHS, validation_data=val_gen)

# ========================
# 🔓 Fine-tuning (descongelar capas superiores)
# ========================
print("🔓 Fine-tuning: descongelando capas superiores de MobileNetV2...")

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False  # solo afina últimas 30 capas

# Recompilar con tasa de aprendizaje más baja
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print("🔁 Entrenando modelo (fase 2, fine-tuning)...")
history_fine = model.fit(train_gen, epochs=FINE_TUNE_EPOCHS, validation_data=val_gen)

# ========================
# 📈 Gráfica de accuracy
# ========================
print("📊 Generando gráfica de accuracy...")
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
epochs_range = range(NUM_EPOCHS + FINE_TUNE_EPOCHS)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label='Entrenamiento')
plt.plot(epochs_range, val_acc, label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('modelos/accuracy_plot.png')
plt.show()

# ========================
# 💾 Guardar modelo y clases
# ========================
os.makedirs("modelos", exist_ok=True)

# model.save(MODELO_PATH)

# New – use native Keras format:
model.save('modelos/model_dermatologico.keras')

print(f"✅ Modelo guardado en: {MODELO_PATH}")

# Guardar clases correctamente
with open(CLASES_PATH, 'w', encoding='utf-8') as f:
    json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=4)

print(f"✅ Clases guardadas en: {CLASES_PATH}")
print("✅ Entrenamiento completo.")
