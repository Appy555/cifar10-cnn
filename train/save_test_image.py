import os
import numpy as np
import tensorflow as tf
from PIL import Image


# CIFAR-10 dataset load karo
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# CIFAR-10 classes
classes = [
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


# Test dataset ki first image
image = x_test[0]
label = int(y_test[0][0])

print("Actual class:", classes[label])


# Project root find karo
project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# test_images folder banao
test_dir = os.path.join(project_root, "test_images")
os.makedirs(test_dir, exist_ok=True)


# Image save karo
image_path = os.path.join(
    test_dir,
    f"test_{classes[label]}.png"
)


Image.fromarray(image).save(image_path)


print("Test image saved at:")
print(image_path)