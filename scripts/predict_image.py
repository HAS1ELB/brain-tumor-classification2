import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import sys
import os

# Load the model
MODEL_PATH = "models/final_model.h5"
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to preprocess the image
def preprocess_image(image_path):
    """
    Preprocess the image for prediction.
    Args:
        image_path (str): Path to the image file.
    Returns:
        np.ndarray: Preprocessed image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Error: Image not found. Please check the path.")
    img = cv2.resize(img, (128, 128))  # Resize image to 128x128
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict function
def predict_image(image_path):
    """
    Predict the class of a given image.
    Args:
        image_path (str): Path to the image file.
    """
    try:
        img = preprocess_image(image_path)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        print(f"Predicted Class: {class_names[predicted_class]} \nConfidence: {confidence:.2f}")
    except Exception as e:
        print(f"Error: {e}")

# Main function to take image path as input
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("Error: Image path does not exist.")
        sys.exit(1)

    predict_image(image_path)
