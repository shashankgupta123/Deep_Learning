import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5

MODEL_FILE = "cifar10_efficientnetv2.keras"
METRICS_FILE = "training_metrics.json"
GRAPH_FILE = "training_history.png"


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
    "truck"
]


# Reproducibility
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD CIFAR-10 DATASET
# ============================================================

print("Loading CIFAR-10 dataset...")

(x_train, y_train), (x_test, y_test) = (
    keras.datasets.cifar10.load_data()
)

y_train = y_train.squeeze()
y_test = y_test.squeeze()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)

# ============================================================
# CREATE TRAINING / VALIDATION SPLIT
# ============================================================

validation_size = 5000

x_val = x_train[-validation_size:]
y_val = y_train[-validation_size:]

x_train = x_train[:-validation_size]
y_train = y_train[:-validation_size]

print("\nAfter validation split:")
print("Training:", x_train.shape)
print("Validation:", x_val.shape)
print("Testing:", x_test.shape)


# ============================================================
# TF.DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def prepare_image(image, label):

    image = tf.cast(
        image,
        tf.float32
    )

    # Resize CIFAR-10 image from 32x32 to 224x224
    image = tf.image.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    return image, label


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (x_train, y_train)
    )
    .shuffle(
        10000,
        seed=SEED
    )
    .map(
        prepare_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

validation_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (x_val, y_val)
    )
    .map(
        prepare_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (x_test, y_test)
    )
    .map(
        prepare_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.1
        ),

        layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08
        ),

        layers.RandomContrast(
            0.1
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNETV2B0
# ============================================================

print("\nLoading pretrained EfficientNetV2B0...")

base_model = (
    tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        include_preprocessing=True
    )
)


# Freeze pretrained model initially
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = keras.Input(
    shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    10,
    activation="softmax"
)(x)


model = keras.Model(
    inputs,
    outputs,
    name="CIFAR10_EfficientNetV2B0"
)


# ============================================================
# COMPILE STAGE 1
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_FILE,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    mode="max",
    verbose=1
)


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr
]

# ============================================================
# STAGE 1 - TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n====================================")
print("STAGE 1: TRANSFER LEARNING")
print("====================================")

history1 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=INITIAL_EPOCHS,

    callbacks=callbacks
)


# ============================================================
# STAGE 2 - FINE-TUNING
# ============================================================

print("\n====================================")
print("STAGE 2: FINE-TUNING")
print("====================================")


base_model.trainable = True


# Keep most layers frozen
# Fine-tune only the last portion
fine_tune_at = len(
    base_model.layers
) - 40


for layer in base_model.layers[
    :fine_tune_at
]:

    layer.trainable = False


# Keep BatchNormalization layers frozen
# for more stable transfer learning

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)

history2 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=callbacks
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")

model = keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n====================================")
print("FINAL TEST EVALUATION")
print("====================================")


test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print(
    f"\nFinal Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Final Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

accuracy = (
    history1.history["accuracy"]
    +
    history2.history["accuracy"]
)

val_accuracy = (
    history1.history["val_accuracy"]
    +
    history2.history["val_accuracy"]
)

loss = (
    history1.history["loss"]
    +
    history2.history["loss"]
)

val_loss = (
    history1.history["val_loss"]
    +
    history2.history["val_loss"]
)


# ============================================================
# SAVE TRAINING METRICS
# ============================================================

metrics = {

    "model":
        "EfficientNetV2B0",

    "dataset":
        "CIFAR-10",

    "classes":
        CLASS_NAMES,

    "training_samples":
        len(x_train),

    "validation_samples":
        len(x_val),

    "test_samples":
        len(x_test),

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(test_accuracy),

    "test_accuracy_percent":
        round(
            float(test_accuracy) * 100,
            2
        )
}


with open(
    METRICS_FILE,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

print(
    f"\nMetrics saved to: "
    f"{METRICS_FILE}"
)


# ============================================================
# CREATE TRAINING GRAPH
# ============================================================

epochs_range = range(
    1,
    len(accuracy) + 1
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    epochs_range,
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracy,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(
    "CIFAR-10 EfficientNetV2B0 Accuracy"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    GRAPH_FILE,
    dpi=150
)

plt.close()


print(
    f"Training graph saved to: "
    f"{GRAPH_FILE}"
)


# ============================================================
# TEST SAMPLE PREDICTIONS
# ============================================================

print("\n====================================")
print("SAMPLE PREDICTIONS")
print("====================================")


sample_images = x_test[:10]

resized_images = tf.image.resize(
    sample_images,
    (
        IMAGE_SIZE,
        IMAGE_SIZE
    )
)


predictions = model.predict(
    resized_images,
    verbose=0
)


for index in range(10):

    predicted_index = int(
        np.argmax(
            predictions[index]
        )
    )

    confidence = float(
        np.max(
            predictions[index]
        )
    )

    actual_class = CLASS_NAMES[
        int(y_test[index])
    ]

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    print(
        f"Image {index + 1}: "
        f"Actual = {actual_class}, "
        f"Predicted = {predicted_class}, "
        f"Confidence = "
        f"{confidence * 100:.2f}%"
    )


print(
    "\nTraining and evaluation complete."
)

print(
    f"Best model saved as: "
    f"{MODEL_FILE}"
)
