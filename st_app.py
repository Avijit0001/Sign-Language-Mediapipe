import streamlit as st
import cv2

# Title of the app
st.title("Webcam Feed")

# Initialize state for webcam and captured image
if 'is_webcam_on' not in st.session_state:
    st.session_state.is_webcam_on = False
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None

# Button to start/stop the webcam
if st.button("Toggle Webcam", key="toggle_webcam"):
    st.session_state.is_webcam_on = not st.session_state.is_webcam_on

# If the webcam is on, display the video feed
if st.session_state.is_webcam_on:
    # Create a placeholder for the webcam feed
    video_placeholder = st.empty()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        while st.session_state.is_webcam_on:
            ret, frame = cap.read()  # Capture frame from webcam
            if not ret:
                st.error("Error: Could not read frame.")
                break
            
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the webcam frame in the Streamlit app
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Button to capture the image with a unique key
            if st.button("Capture", key="capture_button"):
                st.session_state.captured_image = frame_rgb  # Store the captured image
                st.write("Image Captured!")
                st.session_state.is_webcam_on = False  # Turn off the webcam

        # Cleanup: Release the webcam when done
        cap.release()
        video_placeholder.empty()
        st.write("Webcam stopped.")

# Show the captured image if available
if st.session_state.captured_image is not None:
    st.image(st.session_state.captured_image, caption="Captured Image", channels="RGB", use_column_width=True)
