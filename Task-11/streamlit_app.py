import streamlit as st
import requests
 
API_BASE = "http://127.0.0.1:5000"
 
st.set_page_config(
    page_title="MNIST Digit Classifier",
    layout="centered"
)
 
st.title("MNIST Handwritten Digit Classifier")
st.write(
    "Upload an image containing one handwritten digit. "
    "The Streamlit UI sends the image to the Flask deep learning API."
)
 
# Show backend health in the sidebar
try:
    health = requests.get(f"{API_BASE}/health", timeout=2)
    if health.ok:
        st.sidebar.success("Flask API: Healthy")
    else:
        st.sidebar.warning("Flask API responded, but health check failed")
except requests.RequestException:
    st.sidebar.error("Flask API: Unavailable")
 
uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"]
)
 
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Uploaded image", width=250)
 
    if st.button("Predict Digit", type="primary"):
        try:
            files = {
                "image": (
                    uploaded_file.name,
                    image_bytes,
                    uploaded_file.type or "image/png"
                )
            }
 
            response = requests.post(
                f"{API_BASE}/predict",
                files=files,
                timeout=30
            )
 
            if response.ok:
                result = response.json()
                predicted_digit = result["predicted_digit"]
                confidence = result["confidence"]
 
                st.success(f"Predicted Digit: {predicted_digit}")
                st.metric("Confidence", f"{confidence:.4f}")
            else:
                st.error(
                    f"Prediction failed ({response.status_code}): "
                    f"{response.text}"
                )
 
        except requests.RequestException as exc:
            st.error(f"Could not contact Flask API: {exc}")
