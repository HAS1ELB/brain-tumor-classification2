import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Définir les chemins dynamiquement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "final_model.keras")

# Charger le modèle
print(f"📂 Chargement du modèle depuis : {os.path.abspath(MODEL_PATH)}")
try:
    model = load_model(MODEL_PATH)
    print("✅ Modèle chargé avec succès.")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Modèle non trouvé : {os.path.abspath(MODEL_PATH)}")
except Exception as e:
    raise Exception(f"❌ Erreur lors du chargement du modèle : {e}")

# Classes du modèle
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Fonction de prétraitement de l'image
def preprocess_image(image_path):
    """
    Prétraite l'image pour la prédiction.
    Args:
        image_path (str): Chemin vers le fichier image.
    Returns:
        np.ndarray: Image prétraitée.
    """
    print(f"🖼️ Chargement et prétraitement de l'image : {os.path.abspath(image_path)}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Erreur : Impossible de charger l'image. Chemin : {os.path.abspath(image_path)}")
    img = cv2.resize(img, (128, 128))  # Redimensionner en 128x128
    img = img / 255.0  # Normaliser les pixels [0, 1]
    img = np.expand_dims(img, axis=0)  # Ajouter une dimension batch
    return img

# Fonction de prédiction
def predict_image(image_path):
    """
    Prédit la classe d'une image donnée.
    Args:
        image_path (str): Chemin vers le fichier image.
    """
    try:
        img = preprocess_image(image_path)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        print(f"✅ Classe Prédite : {class_names[predicted_class]}")
        print(f"📊 Confiance : {confidence:.2f}")
    except ValueError as ve:
        print(f"❌ Erreur de prétraitement : {ve}")
    except Exception as e:
        print(f"❌ Erreur générale : {e}")

# Fonction principale
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Usage : python predict_image.py <chemin_vers_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Vérifier si le chemin existe
    if not os.path.exists(image_path):
        print(f"❌ Erreur : Le chemin de l'image n'existe pas. Chemin fourni : {os.path.abspath(image_path)}")
        sys.exit(1)
    
    predict_image(image_path)
