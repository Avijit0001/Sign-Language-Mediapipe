import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Specify the path to the folder containing subfolders with images
input_folder = "C:\\Movies\\sign language"  # Replace with the path to your main folder
output_size = 128  # Desired size of the output image
padding = 20  # Padding around the bounding box

# Traverse each subfolder and image
for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        # Check if the file is an image (e.g., .jpg, .png)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(subdir, file)
            
            # Load the image
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image to detect hands
            results = hand_detector.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Calculate the bounding box
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])

                    # Calculate width, height, and center
                    width = x_max - x_min
                    height = y_max - y_min
                    size = max(width, height)

                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2

                    # Apply padding and ensure square bounding box
                    x_min_padded = max(0, x_center - size // 2 - padding)
                    y_min_padded = max(0, y_center - size // 2 - padding)
                    x_max_padded = min(image.shape[1], x_center + size // 2 + padding)
                    y_max_padded = min(image.shape[0], y_center + size // 2 + padding)

                    # Extract the hand region
                    hand_region = image[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

                    # Resize the extracted region to the output size
                    resized_hand = cv2.resize(hand_region, (output_size, output_size))

                    # Create a white background
                    white_background = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255

                    # Draw landmarks and connections on the resized image
                    hand_rgb_resized = cv2.cvtColor(resized_hand, cv2.COLOR_BGR2RGB)
                    result_resized = hand_detector.process(hand_rgb_resized)

                    if result_resized.multi_hand_landmarks:
                        for hand_landmarks_resized in result_resized.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                white_background, hand_landmarks_resized, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                            )

                    # Save the processed image, replacing the original
                    cv2.imwrite(file_path, white_background)

# Release resources
hand_detector.close()
