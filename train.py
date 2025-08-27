import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import os
import shutil

# --- Configuration ---
MIN_ACCURACY = 0.97
BASE_MODEL_DIR = 'exports'
MODEL_NAME = 'my_model'

# --- Step 1: Data Preparation ---
print("Loading and preparing dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 2: Model Training ---
print("Training new model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=30, validation_data=(X_test_scaled, y_test), verbose=0)

# --- Step 3: Accuracy Gate Check ---
print("Evaluating new model...")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"New model accuracy: {accuracy*100:.2f}%")

if accuracy >= MIN_ACCURACY:
    print("New model meets accuracy threshold. Proceeding with promotion.")

    # Find the next version number
    current_versions = [int(v) for v in os.listdir(BASE_MODEL_DIR) if v.isdigit() and os.path.isdir(os.path.join(BASE_MODEL_DIR, v))]
    next_version = max(current_versions) + 1 if current_versions else 1
    
    # Define the new model path
    export_path = os.path.join(BASE_MODEL_DIR, str(next_version))
    
    # Save the model using model.export()
    # This creates the complete SavedModel directory structure.
    try:
        model.export(export_path)
        print(f"Model successfully promoted and saved to: {export_path}")
    except Exception as e:
        print(f"Error saving model with model.export(): {e}")
        
    # Optional: Clean up older versions to save space (uncomment if needed)
    # for v in current_versions:
    #     if v < next_version:
    #         shutil.rmtree(os.path.join(BASE_MODEL_DIR, str(v)))
    #         print(f"Removed older version: {os.path.join(BASE_MODEL_DIR, str(v))}")
else:
    print("New model failed accuracy check. Keeping the previous version.")