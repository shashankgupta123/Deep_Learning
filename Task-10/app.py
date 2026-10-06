from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

MODEL_PATH = "digit_model.keras"

print("Loading deep learning model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "MNIST Deep Learning API",
        "status": "running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert("L")
        image = image.resize((28, 28))

        image_array = np.array(image).astype("float32") / 255.0

        # MNIST uses light digits on a dark background.
        # Invert common white-background handwritten images.
        if image_array.mean() > 0.5:
            image_array = 1.0 - image_array

        image_array = np.expand_dims(image_array, axis=(0, -1))

        predictions = model.predict(image_array, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 4)
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
