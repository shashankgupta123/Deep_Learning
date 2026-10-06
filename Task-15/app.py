from flask import Flask, jsonify, request
import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "cifar10_efficientnetv2.keras"
)
IMAGE_SIZE = 224

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

print("Loading CIFAR-10 EfficientNetV2B0 classifier model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    # EfficientNetV2B0 was saved with include_preprocessing=True.
    # Therefore inference input should remain in the 0-255 pixel range.
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "application": "CIFAR-10 EfficientNetV2B0 Image Classification API",
            "status": "running",
            "model": "EfficientNetV2B0",
            "input_size": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "classes": CLASS_NAMES,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "model_path": MODEL_PATH,
            "model": "EfficientNetV2B0",
        }
    )


@app.route("/classes", methods=["GET"])
def classes():
    return jsonify({"classes": CLASS_NAMES})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        image_array = preprocess_image(image)

        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        probabilities = {
            CLASS_NAMES[i]: round(float(predictions[i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        top_three_indexes = np.argsort(predictions)[-3:][::-1]
        top_three = [
            {
                "class": CLASS_NAMES[int(i)],
                "confidence": round(float(predictions[int(i)]), 4),
            }
            for i in top_three_indexes
        ]

        return jsonify(
            {
                "predicted_class": CLASS_NAMES[predicted_index],
                "confidence": round(confidence, 4),
                "top_three": top_three,
                "probabilities": probabilities,
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
