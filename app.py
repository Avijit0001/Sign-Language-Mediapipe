from flask import Flask, render_template, request, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Initialize global variables
cap = None
model = None
hand_image_path = 'static/hand_image.jpg'
padding = 20  # Padding around the detected hand

# Load Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Directly load the model at startup using the file path
model_file_path = r'C:\Code_Everything\Git_Project\SignLanguageMnist\model.h5'  # <-- Specify your .h5 file path here
model = tf.keras.models.load_model(model_file_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', cam_status='off', prediction=None)

# Route to start the webcam
@app.route('/toggle_cam', methods=['POST'])
def toggle_cam():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera could not be opened.")
            return render_template('index.html', cam_status='off', prediction="Failed to access camera.")
        return render_template('index.html', cam_status='on', prediction=None)
    else:
        cap.release()
        cap = None
        return render_template('index.html', cam_status='off', prediction=None)

# Route to capture the image and save it
@app.route('/capture', methods=['POST'])
def capture():
    global cap, model
    if cap is not None:
        success, frame = cap.read()  # Capture a single frame
        if success:
            # Convert the frame to RGB for Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Perform hand detection
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Extract bounding box coordinates for the hand
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                    # Add padding to the hand bounding box
                    x_min_padded = max(0, x_min - padding)
                    y_min_padded = max(0, y_min - padding)
                    x_max_padded = min(frame.shape[1], x_max + padding)
                    y_max_padded = min(frame.shape[0], y_max + padding)

                    # Crop the hand region with padding
                    hand_image = frame[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

                    # Save the hand image
                    if hand_image.size > 0:  # Check if the hand was cropped correctly
                        cv2.imwrite(hand_image_path, hand_image)
                        # Preprocess the image for the model
                        preprocessed_image = preprocess_image(hand_image)

                        # Predict with the loaded model
                        prediction = predict(preprocessed_image)
                        return render_template('index.html', captured_image=hand_image_path, prediction=prediction)

    return render_template('index.html', captured_image=None, cam_status='off', prediction=None)

# Function to preprocess the hand image for the model
def preprocess_image(image):
    # Resize the image to match the model input size
    img_resized = cv2.resize(image, (224, 224))  # Assuming the model takes 224x224 images
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class of the image using the model
def predict(image):
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Assuming classification model
    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)
