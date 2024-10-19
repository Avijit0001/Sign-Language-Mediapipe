from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('C:\\Code_EveryThing\\Git_Project\\SignLanguageMnist\\model.h5')

# Class names (A-Z + special cases)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
               'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
               'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Global variables for storing the sentence and current frame
predicted_letters = []

# Function to predict the letter
def predict_sign(frame):
    # Preprocess the frame for the model
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    return class_names[predicted_index]

# Camera feed generator
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert the frame to JPEG format for the webpage
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    ret, frame = cv2.VideoCapture(0).read()
    if not ret:
        return jsonify({"error": "Camera not working"})
    
    # Get the current prediction
    predicted_class = predict_sign(frame)
    
    # Manage the sentence
    if predicted_class == 'del':
        if predicted_letters:
            predicted_letters.pop()
    elif predicted_class == 'space':
        predicted_letters.append(' ')
    elif predicted_class == 'nothing':
        pass  # Do nothing
    else:
        predicted_letters.append(predicted_class)

    return jsonify({
        "predicted_class": predicted_class,
        "sentence": ''.join(predicted_letters)
    })

@app.route('/clear', methods=['POST'])
def clear():
    global predicted_letters
    predicted_letters = []
    return jsonify({"message": "Sentence cleared"})

if __name__ == '__main__':
    app.run(debug=True)
