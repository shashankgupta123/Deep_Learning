from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os


app = Flask(__name__)

MODEL_PATH = "digit_model.keras"


# ---------------------------------------------------------
# Load trained deep learning model
# ---------------------------------------------------------

print("Loading deep learning model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ---------------------------------------------------------
# Image preprocessing function
# ---------------------------------------------------------

def preprocess_image(image):

    # Convert image to grayscale
    image = image.convert("L")

    # Convert PIL image to NumPy array
    img_array = np.array(image)

    # MNIST uses white digits on black background.
    # If uploaded image has a white background,
    # invert it automatically.
    if img_array.mean() > 127:
        img_array = 255 - img_array

    # Remove very weak/background pixels
    img_array[img_array < 30] = 0

    # Find coordinates of actual digit pixels
    coords = np.argwhere(img_array > 30)

    # Check whether a digit was detected
    if coords.size == 0:
        raise ValueError(
            "No digit detected in the uploaded image."
        )

    # Find bounding box of the handwritten digit
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1

    # Crop image around the digit
    cropped = img_array[
        y_min:y_max,
        x_min:x_max
    ]

    # Convert cropped NumPy array back to PIL
    cropped_image = Image.fromarray(cropped.astype(np.uint8))

    # -----------------------------------------------------
    # Resize while preserving aspect ratio
    # -----------------------------------------------------

    width, height = cropped_image.size

    if width > height:

        new_width = 20

        new_height = max(
            1,
            int(height * (20 / width))
        )

    else:

        new_height = 20

        new_width = max(
            1,
            int(width * (20 / height))
        )

    cropped_image = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # -----------------------------------------------------
    # Create MNIST-style 28x28 black canvas
    # -----------------------------------------------------

    canvas = Image.new(
        "L",
        (28, 28),
        color=0
    )

    # Center resized digit
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    canvas.paste(
        cropped_image,
        (x_offset, y_offset)
    )

    # Convert final image to NumPy array
    final_array = np.array(
        canvas
    ).astype("float32")

    # Normalize pixel values from 0-255 to 0-1
    final_array = final_array / 255.0

    # Add batch and channel dimensions
    #
    # From:
    # 28 x 28
    #
    # To:
    # 1 x 28 x 28 x 1
    final_array = np.expand_dims(
        final_array,
        axis=(0, -1)
    )

    return final_array


# ---------------------------------------------------------
# Home route
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "message":
        "MNIST Deep Learning API",

        "status":
        "running"

    })


# ---------------------------------------------------------
# Health check route
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status":
        "healthy",

        "model_loaded":
        True

    })


# ---------------------------------------------------------
# Prediction route
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether image exists in request
    if "image" not in request.files:

        return jsonify({

            "error":
            "No image file provided"

        }), 400


    file = request.files["image"]


    # Check whether file was selected
    if file.filename == "":

        return jsonify({

            "error":
            "No file selected"

        }), 400


    try:

        # Read uploaded image
        image = Image.open(
            io.BytesIO(
                file.read()
            )
        )

        # Preprocess image
        image_array = preprocess_image(
            image
        )

        # Perform deep learning prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Find class with highest probability
        predicted_digit = int(
            np.argmax(
                predictions[0]
            )
        )

        # Find confidence score
        confidence = float(
            np.max(
                predictions[0]
            )
        )

        # Get probability of all digits
        probabilities = {
            str(i):
            round(
                float(predictions[0][i]),
                4
            )

            for i in range(10)
        }

        # Return prediction result
        return jsonify({

            "predicted_digit":
            predicted_digit,

            "confidence":
            round(
                confidence,
                4
            ),

            "probabilities":
            probabilities

        })


    except ValueError as e:

        return jsonify({

            "error":
            str(e)

        }), 400


    except Exception as e:

        return jsonify({

            "error":
            str(e)

        }), 500


# ---------------------------------------------------------
# Start Flask application
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
