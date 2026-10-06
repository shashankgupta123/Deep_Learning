from pathlib import Path

import tensorflow as tf
from PIL import Image

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

output_dir = Path("test_images")
output_dir.mkdir(exist_ok=True)

(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_test = y_test.flatten()

saved = set()

for image, label in zip(x_test, y_test):
    label = int(label)
    if label not in saved:
        class_name = CLASS_NAMES[label]
        output_path = output_dir / f"{class_name}.png"
        Image.fromarray(image).save(output_path)
        print(f"Saved {class_name}: {output_path}")
        saved.add(label)

    if len(saved) == len(CLASS_NAMES):
        break

print("Test image export complete.")
