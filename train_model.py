# train_model.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- 1. Load Your Image Data ---
DATA_DIR = 'MyData'
IMG_SIZE = 64 # Using a smaller size for faster training
CATEGORIES = sorted(os.listdir(DATA_DIR))
NUM_CLASSES = len(CATEGORIES)

data = []
labels = []

print("Loading image data...")
for category_id, category in enumerate(CATEGORIES):
    path = os.path.join(DATA_DIR, category)
    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(image)
            labels.append(category_id)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

print(f"Loaded {len(data)} images.")

# --- 2. Prepare Data for Training ---
# Reshape data and normalize
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
labels = np.array(labels)

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=NUM_CLASSES)

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# --- 3. Build the Neural Network Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax') # Output layer
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. Train the Model ---
print("\nStarting training...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
print("Training finished.")

# --- 5. Save Your New Model ---
model.save('my_model.h5')
print("\nNew model saved as my_model.h5")