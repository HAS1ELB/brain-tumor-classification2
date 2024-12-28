import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_images_and_labels(data_dir, img_size=(128, 128)):
    """
    Load images and their corresponding labels from the specified directory.
    Args:
        data_dir (str): Path to the directory containing subdirectories for each class.
        img_size (tuple): Size to resize each image to (width, height).

    Returns:
        np.array: Array of images.
        np.array: Array of labels.
    """
    images, labels = [], []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

def preprocess_and_save(data_dir, save_path, img_size=(128, 128)):
    """
    Preprocess images and labels, then save them to a compressed .npz file.
    Args:
        data_dir (str): Path to the directory containing subdirectories for each class.
        save_path (str): Path to save the processed data (.npz file).
        img_size (tuple): Size to resize each image to (width, height).
    """
    print(f"Loading data from {data_dir}...")
    X, y = load_images_and_labels(data_dir, img_size)
    print(f"Loaded {len(X)} images.")

    # Normalize images to [0, 1] range
    X = X / 255.0

    # Encode labels to integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Save the processed data
    print(f"Saving processed data to {save_path}...")
    np.savez(save_path, X=X, y=y, classes=encoder.classes_)
    print(f"Data saved successfully at {save_path}.")

if __name__ == "__main__":
    # Paths to training and testing data directories
    train_data_dir = r"C:\Users\HP\Desktop\brain-tumor-classification\data\raw\Training"
    test_data_dir = r"C:\Users\HP\Desktop\brain-tumor-classification\data\raw\Testing"

    # Paths to save preprocessed data
    train_save_path = r"C:\Users\HP\Desktop\brain-tumor-classification\data\processed\training_data.npz"
    test_save_path = r"C:\Users\HP\Desktop\brain-tumor-classification\data\processed\testing_data.npz"

    # Preprocess and save training data
    preprocess_and_save(train_data_dir, train_save_path)

    # Preprocess and save testing data
    preprocess_and_save(test_data_dir, test_save_path)
