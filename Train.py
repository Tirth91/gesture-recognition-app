import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset 
CSV_FILE = r"D:\Gesture\gesture_data.csv"
data = pd.read_csv(CSV_FILE)

# Extract features and labels
X = data.iloc[:, 1:].values  # Landmark values
y = data.iloc[:, 0].values   # Labels

# Ensure X has exactly 188 features
if X.shape[1] != 188:
    print(f"Error: Expected 188 features, but found {X.shape[1]}. Check dataset.")
    exit()

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label classes for future decoding
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(188,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Add EarlyStopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save model
model.save("landmark_model.h5")

print("Model training complete and saved.")
