import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Charger les données de test
data = np.load(r"C:\Users\HP\Desktop\brain-tumor-classification\data\processed\testing_data.npz")
X_test, y_test, classes = data['X'], data['y'], data['classes']

# Chemins des modèles à comparer
model_paths = {
    "Final Model": r"C:\Users\HP\Desktop\brain-tumor-classification\models\final_model.keras",
    "VGG16": r"C:\Users\HP\Desktop\brain-tumor-classification\models\vgg16_model.keras",
    "VGG19": r"C:\Users\HP\Desktop\brain-tumor-classification\models\vgg19_model.keras",
    "ResNet50": r"C:\Users\HP\Desktop\brain-tumor-classification\models\resnet50_model.keras"  # Assurez-vous que ce modèle existe
}

# Initialiser un dictionnaire pour stocker les résultats
results = {}

# Boucle pour charger chaque modèle, faire des prédictions et calculer les métriques
for model_name, model_path in model_paths.items():
    print(f"\n🔍 Évaluation du modèle: {model_name}")
    
    # Charger le modèle
    model = tf.keras.models.load_model(model_path)
    
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
