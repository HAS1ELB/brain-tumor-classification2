# train_model.py - Entraînement du modèle de classification des tumeurs cérébrales

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import os
import datetime

# Définition dynamique des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "training_data.npz")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "final_model.keras")


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
    print("📊 Chargement des données d'entraînement...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Fichier de données non trouvé : {os.path.abspath(data_path)}")
    
    data = np.load(data_path)
    X = data['X']
    y = tf.keras.utils.to_categorical(data['y'], num_classes=len(data['classes']))
    classes = data['classes']
    
    print(f"✅ {len(X)} images chargées. Classes disponibles : {classes}")
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
    print("🛠️ Construction du modèle CNN...")
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
    model.summary()
    return model


def train_model(model, X_train, y_train, save_path, log_dir):
    """
    Entraîne le modèle avec les données fournies.
    Args:
        model (tf.keras.Model): Modèle CNN.
        X_train (numpy array): Données d'entraînement.
        y_train (numpy array): Labels d'entraînement.
        save_path (str): Chemin pour sauvegarder le modèle.
        log_dir (str): Répertoire pour les logs TensorBoard.
    Returns:
        history (History): Historique de l'entraînement.
    """
    print("🚀 Démarrage de l'entraînement du modèle...")

    # Création des répertoires nécessaires
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(X_train, y_train,
                        epochs=30,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping, checkpoint, tensorboard])
    print("✅ Entraînement terminé. Modèle sauvegardé.")
    return history


def main():
    """
    Fonction principale pour charger les données, construire, entraîner et sauvegarder le modèle.
    """
    try:
        # Chargement des données
        X_train, y_train, classes = load_training_data(TRAIN_DATA_PATH)
        
        # Construction du modèle
        model = build_model(X_train.shape[1:], len(classes))
        
        # Entraînement du modèle
        history = train_model(model, X_train, y_train, MODEL_SAVE_PATH, LOG_DIR)
        
        print("🎯 Modèle entraîné et sauvegardé avec succès.")
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")


if __name__ == "__main__":
    main()
