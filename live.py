import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

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
def process_video(video_path=None):
    # Open video file if a path is provided; otherwise, use the webcam
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, make prediction
        if ret:
            predicted_category, confidence_score = predict_frame(frame)
            label = f'{predicted_category}: {confidence_score:.2f}'
            
            # Display the prediction on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Real-time Detection', frame)
        else:
            # Exit loop if video ends
            break

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Use Webcam")
    print("2. Upload Video File")
    
    choice = input("Enter your choice (1/2): ").strip()
    
    if choice == '1':
        print("Using webcam for real-time detection...")
        process_video()  # Use webcam
    elif choice == '2':
        video_path = input("Enter the path to the video file: ").strip()
        if os.path.exists(video_path):
            print(f"Processing video file: {video_path}")
            process_video(video_path)
        else:
            print("Invalid video file path. Please try again.")
    else:
        print("Invalid choice. Exiting.")
