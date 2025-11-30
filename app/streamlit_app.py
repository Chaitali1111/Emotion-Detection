import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.models import load_model
from typing import Tuple

# --------- Paths ---------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "emotion_model.h5"

# --------- Class Labels (in the same order as training folders) ---------
# Adjust order if your training folders were in a different order
CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

# --------- Load Model (cached) ---------
@st.cache_resource
def load_emotion_model():
    model = load_model(MODEL_PATH)
    return model


def preprocess_image(file) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes an uploaded file from Streamlit, returns:
    - image_rgb: for displaying in the UI
    - img_input: preprocessed image ready for model (1, 48, 48, 1)
    """
    # Read file bytes
    file_bytes = np.frombuffer(file.read(), np.uint8)

    # Decode as color image (BGR)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not read the image. Please upload a valid image file.")

    # Convert to RGB for display in Streamlit
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to grayscale for model
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Resize to 48x48 (FER2013 standard)
    img_resized = cv2.resize(img_gray, (48, 48))

    # Normalize
    img_norm = img_resized.astype("float32") / 255.0

    # Reshape to (1, 48, 48, 1) for CNN input
    img_input = np.expand_dims(img_norm, axis=-1)  # (48, 48, 1)
    img_input = np.expand_dims(img_input, axis=0)  # (1, 48, 48, 1)

    return img_rgb, img_input


def predict_emotion(model, img_input: np.ndarray):
    """
    Returns:
    - predicted_label (str)
    - predicted_prob (float)
    - all_probs (dict: label -> prob)
    """
    preds = model.predict(img_input)[0]  # shape: (7,)
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    prob = float(preds[idx])

    all_probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    return label, prob, all_probs


# --------- Streamlit UI ---------
def main():
    st.set_page_config(
        page_title="Emotion Detection",
        page_icon="😊",
        layout="centered",
    )

    st.title("😊 Emotion Detection")
    


    # Load model once
    model = load_emotion_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a face image (PNG/JPG)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            # Preprocess image
            img_rgb, img_input = preprocess_image(uploaded_file)

            # Show image
            st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

            # Predict button
            if st.button("Predict Emotion"):
                with st.spinner("Analyzing emotion..."):
                    label, prob, all_probs = predict_emotion(model, img_input)

                st.success(f"Predicted Emotion: **{label}** ({prob*100:.2f}% confidence)")

                # Show probability distribution
                st.subheader("Emotion Probabilities")
                sorted_probs = dict(
                    sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                )
                st.bar_chart(sorted_probs)

        except Exception as e:
            st.error(f"Error while processing the image: {e}")


if __name__ == "__main__":
    main()