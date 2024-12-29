# 🧠 **Brain Tumor Classification Using Deep Learning**

## 📚 **Description du Projet**

Ce projet vise à classer les tumeurs cérébrales en quatre catégories : **gliome**, **méningiome**, **sans tumeur**, et **pituitaire**, en utilisant des modèles d'apprentissage profond tels que **CNN personnalisé**, **ResNet50**, **VGG16**, et **VGG19**. L'application Streamlit permet aux utilisateurs de charger des images IRM et de recevoir une prédiction de la catégorie de la tumeur.

---

## 🌐 **Application Déployée**

L'application est déployée et accessible publiquement à l'adresse suivante :
👉 **[Brain Tumor Classification App](https://brain-tumor-classification2-neu3hwybgrtnhbasibdmln.streamlit.app/)**

---

## 📂 **Structure du Répertoire**

```
HAS1ELB-brain-tumor-classification2/
├── scripts/
│   ├── train_model.py        # Entraînement du modèle CNN
│   ├── predict_image.py      # Prédiction sur une image unique
│   ├── preprocess_data.py    # Prétraitement des données
│   ├── train_ResNet.py       # Entraînement avec ResNet50
│   ├── evaluate_model.py     # Évaluation du modèle
│   ├── train_vgg16.py        # Entraînement avec VGG16
│   ├── train_vgg19.py        # Entraînement avec VGG19
│   ├── compare_models.py     # Comparaison des performances des modèles
│
├── notebooks/
│   ├── train_model.ipynb
│   ├── compare_models.ipynb
│   ├── preprocess_data.ipynb
│   ├── evaluation_model.ipynb
│   ├── predict_image.ipynb
│
├── data/
│   ├── raw/                  # Données brutes
│   │   ├── Testing/
│   │   ├── Training/
│   ├── processed/            # Données prétraitées
│
├── models/                   # Modèles entraînés sauvegardés
│
├── logs/
│   ├── evaluation_report.txt
│
├── app.py                    # Application Streamlit pour les prédictions
├── requirements.txt          # Dépendances du projet
├── Dockerfile                # Configuration Docker
├── download_processed_data.py
├── .dockerignore
└── README.md
```

---

## 🚀 **Installation**

### Prérequis

- Python 3.8+
- TensorFlow
- Streamlit
- OpenCV
- Scikit-learn

### Étapes

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/HAS1ELB/brain-tumor-classification2
   cd brain-tumor-classification2
   ```
2. Créez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
4. Téléchargez les modèles pré-entraînés (si nécessaire) :
   ```bash
   python download_processed_data.py
   ```

---

## 🛠️ **Utilisation**

### **1. Entraîner un modèle**

```bash
python scripts/train_model.py
```

### **2. Faire une prédiction**

```bash
python scripts/predict_image.py <path_to_image>
```

### **3. Lancer l'application Streamlit**

```bash
streamlit run app.py
```

---

## 📊 **Modèles et Performances**

- **ResNet50**: Accuracy ~98%
- **VGG16**: Accuracy ~94%
- **VGG19**: Accuracy ~93%
- **Custom CNN**: Accuracy ~96%

---

## 📝 **Données**

- Source : [Dataset Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets)

### **Prétraitement :**

- Redimensionnement à **128x128**
- Normalisation des pixels entre 0 et 1
- Encodage des labels

---

## 📈 **Évaluation Et Comparaison des Modèles**

Pour évaluer les performances :

```bash
python scripts/compare_models.py
```

---

## 📦 **Docker**

Pour exécuter le projet avec Docker :

```bash
docker build -t brain-tumor-classification .
docker run -p 8501:8501 brain-tumor-classification
```

---

## 👥 **Project Authors**

- **EL BAHRAOUI HASSAN**
- **Malek Sami**

---

## 🤝 **Contributions**

Les contributions sont les bienvenues ! Ouvrez une **issue** pour signaler un problème ou proposez une **pull request**.
