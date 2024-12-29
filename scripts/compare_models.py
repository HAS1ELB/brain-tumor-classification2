import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Obtenir le chemin absolu du répertoire du script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Charger les données de test
data_path = os.path.join(BASE_DIR, "..", "data", "processed", "testing_data.npz")
print(f"📂 Chargement des données depuis : {os.path.abspath(data_path)}")

try:
    data = np.load(data_path)
    X_test, y_test, classes = data['X'], data['y'], data['classes']
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Fichier non trouvé : {os.path.abspath(data_path)}")

# Chemins des modèles à comparer
model_paths = {
    "Final Model": os.path.join(BASE_DIR, "..", "models", "final_model.keras"),
    "VGG16": os.path.join(BASE_DIR, "..", "models", "vgg16_model.keras"),
    "VGG19": os.path.join(BASE_DIR, "..", "models", "vgg19_model.keras"),
    "ResNet50": os.path.join(BASE_DIR, "..", "models", "resnet50_model.keras")
}

# Initialiser un dictionnaire pour stocker les résultats
results = {}

# Boucle pour charger chaque modèle, faire des prédictions et calculer les métriques
for model_name, model_path in model_paths.items():
    print(f"\n🔍 Évaluation du modèle: {model_name}")
    model_path_abs = os.path.abspath(model_path)
    print(f"📂 Chargement du modèle depuis : {model_path_abs}")
    
    # Charger le modèle
    model = tf.keras.models.load_model(model_path_abs)
    
    # Faire des prédictions
    preds = np.argmax(model.predict(X_test), axis=1)
    
    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, preds)
    results[model_name] = accuracy
    
    # Afficher le rapport de classification
    print(f"📈 Accuracy: {accuracy:.2f}")
    print("\n📝 Classification Report:")
    print(classification_report(y_test, preds, target_names=classes))

# Résumé des résultats
print("\n📊 **Résumé des Performances**")
for model_name, accuracy in results.items():
    print(f"{model_name}: Accuracy = {accuracy:.2f}")
