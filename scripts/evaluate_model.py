# evaluate_model.py - Évaluation du modèle de classification des tumeurs cérébrales

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Définir les chemins dynamiquement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "testing_data.npz")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "final_model.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "logs")

# Créer le répertoire de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_testing_data(data_path):
    """
    Charge les données de test depuis un fichier .npz.
    Args:
        data_path (str): Chemin vers le fichier .npz contenant les données de test.
    Returns:
        X (numpy array): Images normalisées.
        y (numpy array): Labels réels encodés en one-hot.
        classes (list): Liste des noms des classes.
    """
    print(f"📂 Chargement des données de test depuis : {os.path.abspath(data_path)}")
    try:
        data = np.load(data_path)
        X = data['X']  # Normalisation des images
        y = data['y']
        classes = data['classes']
        print(f"✅ Nombre d'images : {len(X)}, Classes : {classes}")
        return X, y, classes
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Fichier non trouvé : {os.path.abspath(data_path)}")

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur les données de test.
    Args:
        model (tf.keras.Model): Modèle chargé pour l'évaluation.
        X_test (numpy array): Données de test.
        y_test (numpy array): Labels réels.
    Returns:
        y_pred (numpy array): Prédictions du modèle.
        accuracy (float): Précision globale sur les données de test.
    """
    print("🧠 Évaluation du modèle...")

    # Ne pas convertir les labels en one-hot
    # Utiliser sparse_categorical_crossentropy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"✅ Précision sur les données de test : {accuracy:.2%}")
    
    # Prédictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return y_pred, accuracy

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    """
    Affiche et sauvegarde la matrice de confusion.
    Args:
        y_true (numpy array): Labels réels.
        y_pred (numpy array): Prédictions.
        classes (list): Noms des classes.
        output_dir (str): Répertoire pour sauvegarder la matrice.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.title("Matrice de Confusion")
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path)
    plt.show()
    print(f"✅ Matrice de confusion sauvegardée dans : {os.path.abspath(output_path)}")

def print_classification_report(y_true, y_pred, classes):
    """
    Affiche un rapport de classification détaillé.
    Args:
        y_true (numpy array): Labels réels.
        y_pred (numpy array): Prédictions.
        classes (list): Noms des classes.
    """
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\n📝 Rapport de classification :\n")
    print(report)

def main():
    """
    Fonction principale pour charger les données de test, évaluer le modèle et afficher les métriques.
    """
    # Charger les données de test
    X_test, y_test, classes = load_testing_data(TEST_DATA_PATH)
    
    # Charger le modèle sauvegardé
    print(f"📂 Chargement du modèle depuis : {os.path.abspath(MODEL_PATH)}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Modèle non trouvé : {os.path.abspath(MODEL_PATH)}")
    
    # Évaluer le modèle
    y_pred, accuracy = evaluate_model(model, X_test, y_test)
    
    # Afficher la matrice de confusion
    plot_confusion_matrix(y_test, y_pred, classes, OUTPUT_DIR)
    
    # Afficher le rapport de classification
    print_classification_report(y_test, y_pred, classes)

if __name__ == "__main__":
    main()
