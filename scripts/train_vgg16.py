import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# Définition des chemins relatifs dynamiquement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "training_data.npz")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "vgg16_model.keras")

# Charger les données prétraitées
data = np.load(TRAIN_DATA_PATH)
X, y, classes = data['X'], data['y'], data['classes']

# Diviser les données
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Charger VGG16 pré-entraîné
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Geler les couches du modèle pré-entraîné
for layer in base_model.layers:
    layer.trainable = False

# Ajouter des couches personnalisées
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Ajouter des Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
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
