from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


model_path = r"C:\\Code_EveryThing\\Git_Project\\SignLanguageMnist\\densenet201model.h5"
model = tf.keras.models.load_model(model_path)


class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


cap = cv2.VideoCapture(0)
hand_image_path = 'static/hand_image.jpg'
padding = 20  


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                    
                    x_min_padded = max(0, x_min - padding)
                    y_min_padded = max(0, y_min - padding)
                    x_max_padded = min(frame.shape[1], x_max + padding)
                    y_max_padded = min(frame.shape[0], y_max + padding)

                    
                    cv2.rectangle(frame, (x_min_padded, y_min_padded), (x_max_padded, y_max_padded), (0, 255, 0), 2)

            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    global hand_image_path
    success, frame = cap.read()  
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                
                width = x_max - x_min
                height = y_max - y_min
                
                
                size = max(width, height)

                
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2

                
                x_min_padded = max(0, x_center - size // 2 - padding)
                y_min_padded = max(0, y_center - size // 2 - padding)
                x_max_padded = min(frame.shape[1], x_center + size // 2 + padding)
                y_max_padded = min(frame.shape[0], y_center + size // 2 + padding)

                
                white_background = np.ones((size + 2 * padding, size + 2 * padding, 3), dtype=np.uint8) * 255

                
                scaled_landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int((landmark.x * frame.shape[1] - x_min_padded) * (white_background.shape[1] / (x_max_padded - x_min_padded)))
                    y = int((landmark.y * frame.shape[0] - y_min_padded) * (white_background.shape[0] / (y_max_padded - y_min_padded)))
                    scaled_landmarks.append((x, y))

                
                for idx, point in enumerate(scaled_landmarks):
                    cv2.circle(white_background, point, 5, (0, 0, 255), -1)  

                connections = mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx, end_idx = connection
                    cv2.line(white_background, scaled_landmarks[start_idx], scaled_landmarks[end_idx], (0, 255, 0), 2)  # Green lines for pipes

                
                cv2.imwrite(hand_image_path, white_background)
                print(f"Image saved with landmarks at {hand_image_path}")
                break  

    return render_template('index.html', captured_image=hand_image_path)



@app.route('/predict', methods=['POST'])
def predict():
    
    hand_image = cv2.imread(hand_image_path)
    
    
    img_resized = cv2.resize(hand_image, (128, 128))  # Assuming the model takes 224x224 images
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]  # Get the predicted class label

    return render_template('index.html', captured_image=hand_image_path, prediction=predicted_class)


@app.route('/')
def index():
    return render_template('index.html', captured_image=None, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
