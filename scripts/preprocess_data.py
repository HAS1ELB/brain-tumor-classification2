import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Définition dynamique des chemins de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

# Fonction pour charger les images et leurs étiquettes
def load_images_and_labels(data_dir, img_size=(128, 128)):
    """
    Charge les images et leurs étiquettes depuis un répertoire donné.
    Args:
        data_dir (str): Chemin vers le répertoire contenant les sous-dossiers de classes.
        img_size (tuple): Taille de redimensionnement (largeur, hauteur).

    Returns:
        np.ndarray: Tableau d'images.
        np.ndarray: Tableau d'étiquettes.
    """
    images, labels = [], []
    print(f"📂 Chargement des images depuis : {os.path.abspath(data_dir)}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Répertoire introuvable : {os.path.abspath(data_dir)}")
    
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            print(f"🔄 Traitement de la classe : {label}")
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"❌ Impossible de charger l'image : {img_path}")
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"⚠️ Erreur lors du traitement de l'image {img_path}: {e}")
    print(f"✅ Chargement terminé : {len(images)} images chargées.")
    return np.array(images), np.array(labels)

# Fonction pour prétraiter et sauvegarder les données
def preprocess_and_save(data_dir, save_path, img_size=(128, 128)):
    """
    Prétraite les images et étiquettes, puis les sauvegarde dans un fichier compressé .npz.
    Args:
        data_dir (str): Répertoire contenant les sous-dossiers de classes.
        save_path (str): Chemin de sauvegarde du fichier .npz.
        img_size (tuple): Taille des images redimensionnées.
    """
    print(f"📊 Prétraitement des données depuis : {os.path.abspath(data_dir)}")
    X, y = load_images_and_labels(data_dir, img_size)
    
    # Normalisation des images
    X = X / 255.0
    
    # Encodage des étiquettes
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    # Sauvegarde des données prétraitées
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Crée le dossier si nécessaire
    print(f"💾 Sauvegarde des données prétraitées dans : {os.path.abspath(save_path)}")
    np.savez_compressed(save_path, X=X, y=y, classes=encoder.classes_)
    print("✅ Données sauvegardées avec succès.")

# Fonction principale
if __name__ == "__main__":
    # Chemins des répertoires et fichiers
    train_data_dir = os.path.join(RAW_DATA_DIR, "Training")
    test_data_dir = os.path.join(RAW_DATA_DIR, "Testing")
    train_save_path = os.path.join(PROCESSED_DATA_DIR, "training_data.npz")
    test_save_path = os.path.join(PROCESSED_DATA_DIR, "testing_data.npz")
    
    # Prétraitement et sauvegarde des données d'entraînement
    preprocess_and_save(train_data_dir, train_save_path)
    
    # Prétraitement et sauvegarde des données de test
    preprocess_and_save(test_data_dir, test_save_path)
