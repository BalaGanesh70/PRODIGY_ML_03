import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess the dataset
def load_dataset(data_dir):
    classes = sorted(os.listdir(data_dir))
    images = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(class_path, filename)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(i)
                
    return np.array(images), np.array(labels)

# Replace with the path to your dataset
dataset_path = "path/to/hand_gestures_dataset"
X, y = load_dataset(dataset_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Display classification report
class_names = sorted(os.listdir(dataset_path))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
