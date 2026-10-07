import os
import tensorflow as tf
from tensorflow.keras import layers, models


# CNN Model
model = models.Sequential([

    # Input image: 32x32 RGB
    layers.Input(shape=(32, 32, 3)),

    # First Convolution layer
    layers.Conv2D(32, (3, 3), activation="relu"),

    # Reduce image size
    layers.MaxPooling2D((2, 2)),

    # Second Convolution layer
    layers.Conv2D(64, (3, 3), activation="relu"),

    # Reduce image size again
    layers.MaxPooling2D((2, 2)),

    # Convert 2D feature maps into 1D
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(128, activation="relu"),

    # 10 CIFAR-10 classes
    layers.Dense(10, activation="softmax")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Show model architecture
model.summary()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "cifar10_model.keras")
model.save(model_path)

print(f"Model saved successfully at: {model_path}")


print("Model saved successfully")