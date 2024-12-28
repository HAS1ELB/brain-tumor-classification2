import tensorflow as tf
from tensorflow.keras.applications import VGG19  # Importer VGG19 au lieu de VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Charger les données prétraitées
data = np.load(r"C:\Users\HP\Desktop\brain-tumor-classification\data\processed\training_data.npz")
X, y, classes = data['X'], data['y'], data['classes']

# Diviser les données
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Charger VGG19 pré-entraîné
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Geler les couches du modèle pré-entraîné
for layer in base_model.layers:
    layer.trainable = False

# Ajouter des couches personnalisées
model = Sequential([
    base_model,
    Flatten(),  # Aplatir les sorties de la dernière couche convolutionnelle
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout pour éviter le surapprentissage
    Dense(len(classes), activation='softmax')  # Couche finale pour les classes
])

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Ajouter des Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=r"C:\Users\HP\Desktop\brain-tumor-classification\models\vgg19_model.keras",  # Renommé pour VGG19
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

# Entraîner le modèle
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=20, 
                    batch_size=32,
                    callbacks=callbacks)


# Visualisation de l'entraînement
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
