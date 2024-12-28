# train_model.py - Entraînement du modèle de classification des tumeurs cérébrales

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os

# Chemins par défaut
TRAIN_DATA_PATH = "../data/processed/training_data.npz"
MODEL_SAVE_PATH = "../models/final_model.keras"
LOG_DIR = "../logs"


def load_training_data(data_path):
    """
    Charge les données d'entraînement depuis un fichier .npz.
    Args:
        data_path (str): Chemin vers le fichier .npz contenant les données.
    Returns:
        X (numpy array): Images normalisées.
        y (numpy array): Labels encodés en one-hot.
        classes (list): Liste des noms des classes.
    """
    print("Chargement des données d'entraînement...")
    data = np.load(data_path)
    X = data['X'] / 255.0  # Normalisation des images
    y = tf.keras.utils.to_categorical(data['y'], num_classes=len(data['classes']))
    classes = data['classes']
    print(f"Nombre d'images : {len(X)}, Classes : {classes}")
    return X, y, classes


def build_model(input_shape, num_classes):
    """
    Construit et retourne un modèle CNN.
    Args:
        input_shape (tuple): Dimension des images en entrée.
        num_classes (int): Nombre de classes pour la sortie.
    Returns:
        model (tf.keras.Model): Modèle CNN compilé.
    """
    print("Construction du modèle CNN...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, save_path, log_dir):
    """
    Entraîne le modèle avec les données fournies.
    Args:
        model (tf.keras.Model): Modèle CNN.
        X_train (numpy array): Données d'entraînement.
        y_train (numpy array): Labels d'entraînement.
        save_path (str): Chemin pour sauvegarder le modèle.
        log_dir (str): Répertoire pour les logs.
    Returns:
        history (History): Historique de l'entraînement.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    print("Démarrage de l'entraînement...")
    history = model.fit(X_train, y_train,
                        epochs=30,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping, checkpoint])
    return history


def main():
    """
    Fonction principale pour charger les données, construire, entraîner et sauvegarder le modèle.
    """
    X_train, y_train, classes = load_training_data(TRAIN_DATA_PATH)
    model = build_model(X_train.shape[1:], len(classes))
    history = train_model(model, X_train, y_train, MODEL_SAVE_PATH, LOG_DIR)
    print("Entraînement terminé. Modèle sauvegardé.")


if __name__ == "__main__":
    main()