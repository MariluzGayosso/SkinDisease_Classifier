import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

# ======================
# 🎨 CONFIGURACIÓN VISUAL
# ======================
st.set_page_config(
    page_title="Clasificador Dermatológico con IA",
    page_icon="🧴",
    layout="centered"
)

st.markdown("""
    <style>
        .title {
            font-size:38px !important;
            text-align:center;
        }
        .subtitle {
            font-size:20px !important;
            color:#555;
            text-align:center;
            margin-bottom:30px;
        }
        .footer {
            font-size:13px;
            color:gray;
            text-align:center;
            margin-top:30px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔬 Clasificador de Enfermedades de la Piel</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sube una imagen y deja que la IA analice la posible condición dermatológica.</div>', unsafe_allow_html=True)

# ======================
# 📦 Cargar modelo
# ======================
try:
    modelo = load_model("modelos/model_dermatologic.h5")
except Exception as e:
    st.error(f"❌ Error cargando el modelo: {e}")
    st.stop()

# ======================
# 🔠 Cargar clases
# ======================
try:
    with open('modelos/clases.json', 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"❌ Error cargando clases: {e}")
    st.stop()

# ======================
# 🏷️ Etiquetas legibles
# ======================
try:
    with open("etiquetas.json", "r", encoding="utf-8") as f:
        etiquetas_legibles = json.load(f)
except:
    etiquetas_legibles = {}

# ======================
# 📤 Subir imagen
# ======================
imagen_subida = st.file_uploader("📷 Sube una imagen en formato JPG o PNG", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    img = Image.open(imagen_subida).convert('RGB')
    
    with st.container():
        st.image(img, caption="🖼️ Imagen cargada", use_column_width=True)
        st.markdown("---")

    # Preprocesamiento
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # 🔍 Predicción
    prediccion = modelo.predict(img_array)[0]
    clase_idx = int(np.argmax(prediccion))
    confianza = prediccion[clase_idx] * 100

    # Mapear índice -> clase
    try:
        clase_codificada = class_indices[str(clase_idx)]
    except KeyError:
        st.error(f"❌ Error: Índice de clase '{clase_idx}' no encontrado.")
        st.stop()

    nombre_legible = etiquetas_legibles.get(clase_codificada, clase_codificada)

    # 📊 Mostrar resultado
    st.success(f"✅ **Resultado:** {nombre_legible} ({confianza:.2f}% de confianza)")

    if confianza < 50:
        st.warning("⚠️Consulta a un dermatólogo.")
    else:
        st.info("ℹ️ Esta predicción no reemplaza un diagnóstico médico profesional.")

    # 📊 Tabla de confianza
    st.subheader("🔢 Confianza por clase:")

    clases = []
    for i in range(len(prediccion)):
        clave = str(i)
        cod = class_indices.get(clave, f"clase_{clave}")
        nombre = etiquetas_legibles.get(cod, cod)
        clases.append(nombre)

    porcentajes = [round(p * 100, 2) for p in prediccion]
    resultado_ordenado = sorted(zip(clases, porcentajes), key=lambda x: x[1], reverse=True)

    for clase, score in resultado_ordenado:
        st.markdown(f"• **{clase}**: {score:.2f}%")

    # 📈 Gráfico de barras
    st.subheader("📈 Visualización de Predicciones:")
    fig, ax = plt.subplots()
    nombres = [r[0] for r in resultado_ordenado]
    valores = [r[1] for r in resultado_ordenado]
    ax.barh(nombres[::-1], valores[::-1], edgecolor='black')
    ax.set_xlabel('Confianza (%)')
    ax.set_title('Distribución de Predicción')
    st.pyplot(fig)

    # 📝 Footer
    st.markdown('<div class="footer">Desarrollado como proyecto universitario • ISC 93</div>', unsafe_allow_html=True)
