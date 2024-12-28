import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown

# Désactiver certains logs TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Telecharger les models
def download_model_from_drive(url, output_path):
    # Assurez-vous que le répertoire existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(output_path):
        print(f"Téléchargement du fichier depuis Google Drive : {output_path}")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"Fichier déjà existant : {output_path}")
        
models_to_download = {
    "vgg16": ("https://drive.google.com/uc?id=1KV9-r8pebUUE3jMl6CTLfB0OmrL_AI_0", "models/vgg16_model.keras"),
    "resnet50": ("https://drive.google.com/uc?id=1j5jnqWqoDRj-ET9dfVFFgcsV5yslUT9I", "models/resnet50_model.keras"),
    "vgg19": ("https://drive.google.com/uc?id=1TUpulZsDJGn0QLzpOchHLCAXo3vUVX5U", "models/vgg19_model.keras"),
    "final_model": ("https://drive.google.com/uc?id=1jYchL2hN-8dbNIhgR67CcC5934iDfX6l", "models/final_model.keras")
}

for model_name, (url, path) in models_to_download.items():
    if not os.path.exists(path):  # Vérifie si le fichier existe déjà
        print(f"Le fichier {model_name} n'existe pas. Téléchargement en cours...")
        download_model_from_drive(url, path)
    else:
        print(f"Le fichier {model_name} existe déjà à l'emplacement {path}.")



# Chemins des modèles à comparer
model_paths = {
    "Final Model": "models/final_model.keras",
    "VGG16": "models/vgg16_model.keras",
    "VGG19": "models/vgg19_model.keras",
    "ResNet50": "models/resnet50_model.keras",
}

# Charger les modèles avec vérification
models = {}
for name, path in model_paths.items():
    try:
        models[name] = tf.keras.models.load_model(path)
    except ValueError as e:
        st.error(f"❌ Erreur lors du chargement du modèle {name} : {str(e)}")
    except Exception as e:
        st.error(f"⚠️ Une erreur inattendue s'est produite pour le modèle {name} : {str(e)}")

# Classes de prédiction
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Prétraitement de l'image
def preprocess_image(image):
    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Interface Streamlit
st.title("🧠 Classification de Tumeurs Cérébrales avec IA")
st.write("Chargez une image d'IRM pour prédire le type de tumeur cérébrale avec plusieurs modèles différents.")

# Upload d'image
uploaded_file = st.file_uploader("Choisissez une image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Image chargée', use_container_width=True)

    if st.button("Prédire avec tous les modèles"):
        # Prétraitement
        preprocessed_image = preprocess_image(image)
        
        # Dictionnaire pour stocker les résultats
        predictions = {}

        # Boucle sur les modèles
        for model_name, model in models.items():
            pred = model.predict(preprocessed_image)
            pred_class = np.argmax(pred, axis=1)[0]
            confidence = np.max(pred)
            predictions[model_name] = (class_names[pred_class], confidence)
        
        # Afficher les résultats
        st.subheader("📊 **Résultats de la Prédiction :**")
        for model_name, (pred_class, confidence) in predictions.items():
            st.write(f"### 🔹 **{model_name}**")
            st.write(f"**Classe prédite:** {pred_class}")
            st.write(f"**Confiance:** {confidence:.2f}")
        
        # Comparaison des modèles
        st.write("### ⚖️ **Comparaison des Modèles**")
        predicted_classes = [value[0] for value in predictions.values()]
        if len(set(predicted_classes)) == 1:
            st.success(f"✅ Tous les modèles prédisent la même classe : **{predicted_classes[0]}**")
        else:
            st.warning("⚠️ Les modèles ont donné des prédictions différentes.")
            st.write("**Résumé des prédictions :**")
            for model_name, (pred_class, _) in predictions.items():
                st.write(f"- **{model_name}** : {pred_class}")
