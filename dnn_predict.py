import numpy as np
from PIL import Image
from dnn_functions import *
from dnn_accuracy import *

def preprocess_image(image_path, target_size):
    """Load, resize, flatten and normalize the image."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)  # e.g., (64, 64)
    image_array = np.array(image)
    image_flatten = image_array.reshape(-1, 1) / 255.0  # flatten and normalize
    return image_flatten

if __name__ == "__main__":
    # === Load the trained model ===
    parameters, network_model, activations = load_model("trained_model.pkl")

    # === Prepare custom image ===
    target_size = (64, 64)  # must match training size
    X_custom = preprocess_image("custom-images/nocat-4.jpg", target_size)

    # === Predict ===
    A_pred = predict(X_custom, parameters, activations)
    y_pred = (A_pred > 0.5).astype(int)

    print(f"\nPrediction output: {A_pred.flatten()[0]:.4f}")
    print(f"Predicted class: {y_pred.flatten()[0]}")
