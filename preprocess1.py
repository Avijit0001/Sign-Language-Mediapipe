import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hand Detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detect_and_crop_hand(image):
    """Detect hand(s) and return cropped hand image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Get bounding box of hand(s)
        h, w, _ = image.shape
        x_min, y_min, x_max, y_max = w, h, 0, 0
        
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
        
        # Add padding around the hand (optional)
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop the image
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image
    else:
        return None

def process_images(input_folder):
    """Process images from input_folder and save cropped hand images overwriting originals."""
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                # Load the image
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                # Detect and crop hand
                cropped_image = detect_and_crop_hand(image)
                
                if cropped_image is not None:
                    # Overwrite the original image with the cropped hand image
                    cv2.imwrite(image_path, cropped_image)
                    print(f"Cropped and saved: {image_path}")
                else:
                    print(f"No hand detected in: {image_path}")

if __name__ == "__main__":
    input_folder = "C:\\Movies\\asl_alphabet_train"  # Replace with your folder path

    # Process the images and overwrite originals
    process_images(input_folder)
