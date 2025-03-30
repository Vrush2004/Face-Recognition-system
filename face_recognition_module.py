import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

TF_ENABLE_ONEDNN_OPTS=0

# Constants
UPLOAD_FOLDER = 'data'  # Folder containing student images
MODEL_PATH = 'face_recognition_googlenet_model.weights.h5'  

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = None

# Load or create the model
def load_existing_model():
    global model
    if os.path.exists(MODEL_PATH):
        print("Loading existing model weights...")
        _, _, num_classes = load_data()  # Get current number of classes
        if num_classes == 0:
            print("No training data found. Skipping model load.")
            return

        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)  # Dynamically adjust num_classes

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        try:
            model.load_weights(MODEL_PATH)  # Load weights safely
            print("Model loaded successfully.")
        except ValueError as e:
            print(f"Warning: Model structure changed. Resetting weights. Error: {e}")
    else:
        print("No existing weights found. Creating a new model...")
        train_data, train_labels, num_classes = load_data()
        if num_classes == 0:
            print("No training data found. Cannot create model.")
            return
        model = create_model(num_classes)

# Create a new model if no saved model exists
def create_model(num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # Use dynamic num_classes

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to load all images and labels
def load_data():
    data = []
    labels = []
    students = os.listdir(UPLOAD_FOLDER)

    for student_id in students:
        student_folder = os.path.join(UPLOAD_FOLDER, student_id)

        if os.path.isdir(student_folder):
            for filename in os.listdir(student_folder):
                img_path = os.path.join(student_folder, filename)

                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue  # Skip if image is invalid

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect face
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face = img[y:y + h, x:x + w]
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = face / 255.0  

                    data.append(face)
                    labels.append(student_id)  # Student ID as label

    if len(data) == 0:
        return None, None, 0  # Return 0 classes if no data is found

    # Convert labels to categorical format
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(np.unique(labels))  # Get number of unique classes dynamically
    labels = to_categorical(labels, num_classes=num_classes)

    data = np.array(data)
    return data, labels, num_classes

# Train model using all data
@app.route('/train', methods=['POST'])
def train():
    try:
        print("Loading training data...")
        train_data, train_labels, num_classes = load_data()

        if train_data is None or train_labels is None or num_classes == 0:
            print("No valid training data found!")
            return jsonify({"error": "No valid training data found!"}), 400

        print(f"Training data shape: {train_data.shape}")
        print(f"Training labels shape: {train_labels.shape}")
        print(f"Number of classes: {num_classes}")

        global model
        if model is None:
            print("Loading existing model...")
            load_existing_model()

        # If new students are added, retrain only the classifier layer
        if model.output_shape[-1] != num_classes:
            print("Updating model with new class size...")
            base_model = Model(inputs=model.input, outputs=model.layers[-3].output)
            x = Dense(1024, activation='relu')(base_model.output)
            predictions = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Reduce batch size to avoid memory issues
        model.fit(train_data, train_labels, epochs=5, batch_size=4, verbose=1)

        # Save model weights instead of entire model to allow incremental training
        weights_filename = "face_recognition_googlenet_model.weights.h5"
        model.save_weights(weights_filename) 


        print("Incremental training completed successfully!")

        return jsonify({"success": True, "message": "Incremental training completed successfully!"})
    
    except Exception as e:
        print(f"Error during incremental training: {e}")
        return jsonify({"error": str(e)}), 500

# Load model at startup if data exists
load_existing_model()