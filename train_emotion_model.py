import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Load and preprocess dataset
# ---------------------------

def load_data(data_dir, target_size=(48, 48)):
    data = []
    labels = []
    emotion_labels = os.listdir(data_dir)

    for emotion in emotion_labels:
        emotion_dir = os.path.join(data_dir, emotion)
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, target_size)
            img = img / 255.0
            data.append(img)
            labels.append(emotion)
    
    data = np.array(data).reshape(-1, 48, 48, 1)
    labels = np.array(labels)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels)

    return data, labels, le.classes_

# Change these paths according to your folder structure
train_dir = "dataset/train"
test_dir = "dataset/test"

print("Loading and preprocessing data...")
x_train, y_train, class_names = load_data(train_dir)
x_test, y_test, _ = load_data(test_dir)
print("Done! Classes:", class_names)

# ---------------------------
# Step 2: Build the CNN model
# ---------------------------

model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# ---------------------------
# Step 3: Train the model
# ---------------------------

print("Training the model...")
history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))

# ---------------------------
# Step 4: Save the model
# ---------------------------

model.save("emotion_model.h5")
print("Model saved as emotion_model.h5")

# ---------------------------
# Step 5: Plot training results
# ---------------------------

plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
