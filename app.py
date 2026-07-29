import pathlib

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (224, 224)
MODEL_PATH = pathlib.Path(__file__).parent / "model.keras"
CLASS_NAMES = ["Formalin-mixed Orange", "Fresh Orange"]

ACCEPTED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif"]

st.set_page_config(page_title="Fresh vs Formalin-mixed Orange Classifier", page_icon="🍊", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"model.keras not found at {MODEL_PATH}. Place it next to app.py.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, pil_image):
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    prob_fresh = float(model.predict(arr, verbose=0)[0][0])
    prob_formalin = 1.0 - prob_fresh

    label = CLASS_NAMES[1] if prob_fresh >= 0.5 else CLASS_NAMES[0]
    return label, prob_fresh * 100, prob_formalin * 100


st.title("🍊 Fresh vs Formalin-mixed Orange Classifier")
st.write("Upload a photo of an orange to check whether it's fresh or formalin-mixed.")

model = load_model()

uploaded_file = st.file_uploader("Upload an orange image", type=ACCEPTED_TYPES)

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        img.load()
    except UnidentifiedImageError:
        st.error("Couldn't read that file as an image. Try a different photo.")
        st.stop()

    st.image(img, width=300, caption="Uploaded image")

    with st.spinner("Classifying..."):
        label, fresh_pct, formalin_pct = predict(model, img)

    st.write(f"### Prediction: **{label}**")
    st.progress(int(round(fresh_pct)), text=f"Fresh: {fresh_pct:.1f}%")
    st.progress(int(round(formalin_pct)), text=f"Formalin-mixed: {formalin_pct:.1f}%")

    if label == CLASS_NAMES[1]:
        st.success("This orange looks fresh.")
    else:
        st.warning("This orange may be formalin-mixed.")
else:
    st.info("Upload a .jpg, .jpeg, .png, .webp, .bmp, .tiff, or .gif image of an orange to get a prediction.")

st.divider()
st.caption("Fresh vs Formalin-mixed Orange | MobileNetV2 transfer learning")