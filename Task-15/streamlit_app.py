import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:5000").rstrip("/")

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered",
)

st.title("CIFAR-10 Image Classification")
st.write(
    "Upload an image and the deep-learning model will classify it into one "
    "of the CIFAR-10 categories."
)

with st.expander("Supported classes"):
    st.write(
        "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck"
    )

# API health status
try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    if health_response.ok:
        st.success("Flask API is available and the model is loaded.")
    else:
        st.warning("Flask API responded but is not healthy.")
except requests.RequestException:
    st.error(f"Flask API is unavailable at {API_URL}")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify Image", type="primary"):
        uploaded_file.seek(0)
        files = {
            "image": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/png",
            )
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=30,
            )

            if response.ok:
                result = response.json()

                st.subheader("Prediction")
                st.metric(
                    "Predicted class",
                    result["predicted_class"],
                )
                st.metric(
                    "Confidence",
                    f"{result['confidence'] * 100:.2f}%",
                )

                st.subheader("Top 3 predictions")
                for item in result["top_three"]:
                    st.write(
                        f"{item['class']}: "
                        f"{item['confidence'] * 100:.2f}%"
                    )

                probabilities = result["probabilities"]
                chart_data = pd.DataFrame(
                    {
                        "Class": list(probabilities.keys()),
                        "Probability": list(probabilities.values()),
                    }
                ).set_index("Class")

                st.subheader("All class probabilities")
                st.bar_chart(chart_data)

            else:
                st.error(response.json().get("error", "Prediction failed"))

        except requests.RequestException as exc:
            st.error(f"Unable to contact Flask API: {exc}")
