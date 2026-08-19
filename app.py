import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from io import BytesIO
import requests

#api
MEDGEMMA_API_URL = "https://stifle-deploy-emoticon.ngrok-free.dev/interpret"

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IA Medical Image Analysis by Mr Z",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 35px;
}

.upload-box {
    padding: 25px;
    border-radius: 15px;
    border: 2px dashed #9ca3af;
    text-align: center;
    margin-bottom: 25px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

.normal {
    background-color: #ecfdf5;
    border: 2px solid #10b981;
}

.pneumonia {
    background-color: #fef2f2;
    border: 2px solid #ef4444;
}

.probability {
    font-size: 34px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #9ca3af;
    margin-top: 50px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FOCAL LOSS
# ============================================================

def FocalLoss(gamma=2.0, alpha=0.25):

    def focal_loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)

        epsilon = tf.keras.backend.epsilon()

        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon
        )

        alpha_t = (
            y_true * alpha
            + (1 - y_true) * (1 - alpha)
        )

        p_t = (
            y_true * y_pred
            + (1 - y_true) * (1 - y_pred)
        )

        loss = (
            -alpha_t
            * tf.pow(1 - p_t, gamma)
            * tf.math.log(p_t)
        )

        return tf.reduce_mean(loss)

    return focal_loss


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "best_densenet121_phase2.keras",
        custom_objects={
            "FocalLoss": FocalLoss
        }
    )

    return model


model = load_model()
# ============================================================
# MODÈLE INTERMÉDIAIRE POUR GRAD-CAM
# ============================================================

last_conv_output = model.get_layer(
    "global_average_pooling2d"
).input

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        last_conv_output,
        model.output
    ]
)


# ============================================================
# FONCTION GRAD-CAM
# ============================================================

def generate_gradcam(
    image,
    model,
    grad_model,
    threshold=0.75
):

    # Prétraitement
    image_resized = image.resize((224, 224))

    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_preprocessed = preprocess_input(
        image_array
    )

    # Calcul des gradients
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_preprocessed
        )

        pneumonia_probability = predictions[:, 0]

        if float(pneumonia_probability[0]) >= threshold:

            predicted_class = "PNEUMONIA"
            class_score = pneumonia_probability

        else:

            predicted_class = "NORMAL"
            class_score = 1.0 - pneumonia_probability

    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    heatmap = heatmap.numpy()

    return heatmap


# ============================================================
# INTERFACE
# ============================================================

st.markdown(
    '<div class="title">🫁 IA Medical Image Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Détection automatique de pneumonie'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="upload-box">'
    '<h3>📤 Importer une image médicale</h3>'
    '<p>Veuillez importer une image pour commencer l’analyse.</p>'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload image please",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# SI AUCUNE IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "📷 Veuillez importer une image médicale "
        "pour lancer l’analyse."
    )

    st.markdown(
        '<div class="footer">'
        'Powered by DenseNet121 • AI Medical Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# LECTURE IMAGE
# ============================================================

try:

    image = Image.open(uploaded_file).convert("RGB")

except Exception:

    st.error(
        "❌ Impossible de lire cette image. "
        "Veuillez importer une image JPG ou PNG valide."
    )

    st.stop()


# ============================================================
# AFFICHAGE IMAGE
# ============================================================

st.subheader("🖼️ Image sélectionnée")

st.image(
    image,
    use_container_width=True
)


# ============================================================
# PRÉTRAITEMENT
# ============================================================

image_resized = image.resize((224, 224))

image_array = np.array(
    image_resized,
    dtype=np.float32
)

image_array = np.expand_dims(
    image_array,
    axis=0
)

image_preprocessed = preprocess_input(
    image_array
)


# ============================================================
# PRÉDICTION
# ============================================================

with st.spinner("🧠 Analyse de l'image en cours..."):

    try:

        prediction = model.predict(
            image_preprocessed,
            verbose=0
        )

        probability = float(
            prediction[0][0]
        )

    except Exception as e:

        st.error(
            "❌ Une erreur est survenue pendant "
            "l'analyse de l'image."
        )

        st.stop()


# ============================================================
# SEUIL OPTIMAL
# ============================================================

OPTIMAL_THRESHOLD = 0.75


# ============================================================
# RÉSULTAT
# ============================================================

st.subheader("📊 Résultat de l'analyse")

probability_percent = probability * 100


st.markdown(
    f"""
    <div class="probability">
        {probability_percent:.2f}%
    </div>
    """,
    unsafe_allow_html=True
)


if probability >= OPTIMAL_THRESHOLD:

    st.markdown(
        """
        <div class="result-box pneumonia">
            <h2>🔴 PNEUMONIA</h2>
            <p>
                Le modèle détecte une probabilité élevée
                de pneumonie.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="result-box normal">
            <h2>🟢 NORMAL</h2>
            <p>
                Le modèle ne détecte pas une probabilité
                élevée de pneumonie.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write(
    f"**Seuil de décision : {OPTIMAL_THRESHOLD:.0%}**"
)


# ============================================================
# GRAD-CAM
# ============================================================

st.subheader("🔥 Explication Grad-CAM")

with st.spinner("Génération de la carte d'attention..."):

    heatmap = generate_gradcam(
        image,
        model,
        grad_model,
        threshold=OPTIMAL_THRESHOLD
    )

# Affichage de la superposition
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 7))

ax.imshow(image)

ax.imshow(
    heatmap,
    cmap="jet",
    alpha=0.40,
    extent=(0, image.width, image.height, 0),
    interpolation="bilinear"
)

ax.axis("off")

st.pyplot(fig)

# ============================================================
# CONVERSION GRAD-CAM EN IMAGE POUR L'API
# ============================================================

gradcam_buffer = BytesIO()

fig.savefig(
    gradcam_buffer,
    format="png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=200
)

gradcam_buffer.seek(0)

gradcam_image = Image.open(
    gradcam_buffer
).convert("RGB")

plt.close(fig)

st.caption(
    "La Grad-CAM indique les zones de l'image "
    "ayant le plus contribué à la décision du modèle."
)



# ============================================================
# APPEL API MEDGEMMA
# ============================================================

st.subheader("🧠 Interprétation VLM")

predicted_class = (
    "PNEUMONIA"
    if probability >= OPTIMAL_THRESHOLD
    else "NORMAL"
)

# Convertir la radio originale en mémoire
xray_buffer = BytesIO()
image.save(
    xray_buffer,
    format="PNG"
)
xray_buffer.seek(0)

# Convertir la Grad-CAM en mémoire
gradcam_api_buffer = BytesIO()
gradcam_image.save(
    gradcam_api_buffer,
    format="PNG"
)
gradcam_api_buffer.seek(0)

files = {
    "xray": (
        "xray.png",
        xray_buffer,
        "image/png"
    ),
    "gradcam": (
        "gradcam.png",
        gradcam_api_buffer,
        "image/png"
    )
}

data = {
    "predicted_class": predicted_class,
    "pneumonia_probability": probability
}

headers = {
    "ngrok-skip-browser-warning": "true"
}

with st.spinner("Interprétation MedGemma en cours..."):

    try:

        response = requests.post(
            MEDGEMMA_API_URL,
            files=files,
            data=data,
            headers=headers,
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        interpretation = result[
            "interpretation"
        ]

        st.markdown(
            f"""
            <div style="
                padding: 20px;
                border-radius: 12px;
                background-color: #f8fafc;
                border: 1px solid #e5e7eb;
                line-height: 1.7;
            ">
                {interpretation.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True
        )

    except requests.exceptions.Timeout:

        st.warning(
            "⚠️ L'interprétation VLM prend trop de temps. "
            "Le serveur MedGemma est peut-être temporairement indisponible."
        )

    except requests.exceptions.ConnectionError:

        st.warning(
            "⚠️ Le service MedGemma n'est pas disponible. "
            "La prédiction DenseNet et la Grad-CAM restent accessibles."
        )

    except requests.exceptions.HTTPError as e:

        st.warning(
            "⚠️ Le service MedGemma a rencontré une erreur lors de l'analyse."
        )

        st.caption(str(e))

    except Exception as e:

        st.warning(
            "⚠️ L'interprétation VLM n'a pas pu être générée. "
            "La classification et la Grad-CAM restent disponibles."
        )

        st.caption(str(e))




# ============================================================
# RÉSUMÉ POUR L'INTERPRÉTATION
# ============================================================

if predicted_class == "PNEUMONIA":
    class_probability = probability
else:
    class_probability = 1.0 - probability

st.markdown(
    f"""
    **Classe prédite :** {predicted_class}  
    **Confiance du modèle :** {class_probability * 100:.2f}%  
    **P(PNEUMONIA) :** {probability * 100:.2f}%
    """
)







# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        DenseNet121 • Deep Learning • Medical Image Classification
        <br>
        Ce système est un outil d'aide à l'analyse et ne remplace pas
        l'avis d'un professionnel de santé.
        <br>
        Made by Hamza Kechbal.
    </div>
    """,
    unsafe_allow_html=True
)
