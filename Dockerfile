# Utiliser l'image Python de base
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers du projet dans le conteneur
COPY . /app/

# Vérifiez que le répertoire models est copié correctement
COPY models/vgg16_model.keras /models/vgg16_model.keras
COPY models/final_model.keras /models/final_model.keras
COPY models/vgg19_model.keras /models/vgg19_model.keras
COPY models/resnet50_model.keras /models/resnet50_model.keras

# Exposer le port pour Streamlit
EXPOSE 8501

# Commande par défaut pour exécuter l'application
CMD ["streamlit", "run", "app.py"]
