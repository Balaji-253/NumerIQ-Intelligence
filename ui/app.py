import io
import os

import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a digit from 0 to 9 or upload an image.")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Create tabs
tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

image = None

# Drawing interface
with tab1:
    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
        return_image_data=True
    )

    # Safely access image data
    if canvas.image_data is not None:
        image = Image.fromarray(
            canvas.image_data.astype("uint8"),
            mode="RGBA"
        ).convert("RGB")

# Upload interface
with tab2:
    uploaded_file = st.file_uploader(
        "Upload handwritten digit",
        type=["png", "jpg", "jpeg", "webp", "bmp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).copy()
        st.image(image, caption="Uploaded Image", width=220)

# Prediction button
if st.button("Recognize Digit", type="primary"):

    if image is None:
        st.warning("Please draw or upload a digit first.")

    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        try:
            response = requests.post(
                f"{API_URL}/predict",
                files={
                    "file": (
                        "digit.png",
                        buffer.getvalue(),
                        "image/png"
                    )
                },
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            st.success(
                f"Predicted Digit: {result['digit']}"
            )

            st.metric(
                "Prediction Confidence",
                f"{result['confidence']:.2%}"
            )

            st.subheader("Top 3 Predictions")

            for item in result["top3"]:
                st.progress(
                    item["probability"],
                    text=(
                        f"Digit {item['digit']}: "
                        f"{item['probability']:.2%}"
                    )
                )

        except requests.exceptions.RequestException as error:
            st.error(f"API Connection Error: {error}")
            st.info("Make sure FastAPI is running on port 8000.")