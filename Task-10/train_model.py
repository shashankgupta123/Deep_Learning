import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add a channel dimension: (28, 28) -> (28, 28, 1)
x_train = x_train[..., None]
x_test = x_test[..., None]

# Build a simple Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training model...")

model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1
)

# Evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save("digit_model.keras")
print("Model saved successfully as digit_model.keras")

