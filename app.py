import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
from tempfile import NamedTemporaryFile

# Define categories (same order as used during training)
categories = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'NormalVideos',
              'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

# Load the saved model
model = load_model('resnet_model.h5')

# Function to preprocess a single frame
def preprocess_frame(frame, target_size=(64, 64)):
    frame_resized = cv2.resize(frame, target_size)
    frame_array = img_to_array(frame_resized)
    frame_array = frame_array / 255.0  # Rescale to [0, 1]
    frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension
    return frame_array

# Function to make a prediction
def predict_frame(frame):
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(preprocessed_frame)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_category = categories[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index]
    return predicted_category, confidence_score

# Function to process video
def process_video(video_source, use_webcam=False):
    stframe = st.empty()
    cap = cv2.VideoCapture(0 if use_webcam else video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prediction
        predicted_category, confidence_score = predict_frame(frame)
        label = f'{predicted_category}: {confidence_score:.2f}'
        
        # Add prediction to frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert for Streamlit

        # Display frame in Streamlit
        stframe.image(frame, channels="RGB")

    cap.release()

# Streamlit app interface
def main():
    st.title("Real-Time Video Classification")
    st.write("Choose between Webcam or Video Upload to detect video categories in real-time.")

    # Sidebar for navigation
    choice = st.sidebar.selectbox("Select Input Method", ["Webcam", "Upload Video"])

    if choice == "Webcam":
        st.write("Using webcam for real-time detection. Press 'q' to quit.")
        if st.button("Start Webcam"):
            process_video(None, use_webcam=True)

    elif choice == "Upload Video":
        video_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
        if video_file is not None:
            # Save uploaded video temporarily
            with NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(video_file.read())
                temp_video_path = temp_file.name
            
            st.write("Processing uploaded video...")
            process_video(temp_video_path)

def new_func(__name__, main):
    if __name__ == "__main__":
        main()

