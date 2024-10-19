import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model("C:\\Code_EveryThing\\Git_Project\\SignLanguageMnist\\model.h5")

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def load_and_preprocess_image(img):
    img = img.resize((128, 128))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def classify_image(img):
    img = load_and_preprocess_image(img)
    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0])
    return np.argmax(score), 100 * np.max(score)

st.title("Image Classification App")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    class_id, confidence = classify_image(img)
    st.write(f"Prediction: {class_names[class_id]}")