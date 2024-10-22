from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Model Path 
model_path = r"C:\Code_Everything\Git_Project\SignLanguageMnist\model.h5"
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Global variables
cap = cv2.VideoCapture(0)
hand_image_path = 'static/hand_image.jpg'
padding = 20  # Padding around the detected hand

# Stream the webcam feed continuously
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert the frame to RGB for Mediapipe hand detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            # If hands are detected, draw landmarks and capture the hand region
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                    # Add padding to the hand bounding box
                    x_min_padded = max(0, x_min - padding)
                    y_min_padded = max(0, y_min - padding)
                    x_max_padded = min(frame.shape[1], x_max + padding)
                    y_max_padded = min(frame.shape[0], y_max + padding)

                    # Draw a rectangle around the hand
                    cv2.rectangle(frame, (x_min_padded, y_min_padded), (x_max_padded, y_max_padded), (0, 255, 0), 2)

            # Encode the frame to be sent to the UI
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Capture only the hand from the live feed
@app.route('/capture', methods=['POST'])
def capture():
    global hand_image_path
    success, frame = cap.read()  # Capture a frame from the webcam
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                # Add padding to the hand bounding box
                x_min_padded = max(0, x_min - padding)
                y_min_padded = max(0, y_min - padding)
                x_max_padded = min(frame.shape[1], x_max + padding)
                y_max_padded = min(frame.shape[0], y_max + padding)

                # Crop the hand region
                hand_image = frame[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

                if hand_image.size > 0:
                    cv2.imwrite(hand_image_path, hand_image)
                    break  # Save only the first detected hand

    return render_template('index.html', captured_image=hand_image_path)

# Predict the class of the captured hand image
@app.route('/predict', methods=['POST'])
def predict():
    # Load the captured image
    hand_image = cv2.imread(hand_image_path)
    
    # Preprocess the image
    img_resized = cv2.resize(hand_image, (128, 128))  # Assuming the model takes 224x224 images
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict with the model
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]  # Get the predicted class label

    return render_template('index.html', captured_image=hand_image_path, prediction=predicted_class)

# Main route for rendering the UI
@app.route('/')
def index():
    return render_template('index.html', captured_image=None, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
