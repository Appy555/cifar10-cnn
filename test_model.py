import tensorflow as tf
import numpy as np
from PIL import Image


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


# Model load karo
model = tf.keras.models.load_model(
    "models/cifar10_model.keras"
)


# Same image load karo
image = Image.open(
    "test_images/test_cat.png"
)

# RGB mein convert
image = image.convert("RGB")

# 32x32 resize
image = image.resize((32, 32))

# NumPy array
image_array = np.array(image)

# Normalize
image_array = image_array / 255.0

# Batch dimension add
image_array = np.expand_dims(
    image_array,
    axis=0
)


# Prediction
predictions = model.predict(
    image_array,
    verbose=0
)


# Highest probability
predicted_index = np.argmax(predictions[0])

print("Actual class: cat")
print(
    "Predicted class:",
    classes[predicted_index]
)

print(
    "Confidence:",
    predictions[0][predicted_index] * 100
)