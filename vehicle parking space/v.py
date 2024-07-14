import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the pre-trained LSTM model
model = load_model('lstm_model.h5')

# Load the pre-trained CNN model for feature extraction
cnn_model = load_model('cnn_model.h5')

# Define the video capture device (e.g. webcam)
cap = cv2.VideoCapture(0)

# Define the frame size and frame rate
frame_size = (640, 480)
frame_rate = 30

# Define the buffer size for storing frames
buffer_size = 30

# Initialize the frame buffer
frame_buffer = []

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    
    # Resize the frame to the desired size
    frame = cv2.resize(frame, frame_size)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Extract features from the frame using the CNN model
    features = cnn_model.predict(gray.reshape(1, frame_size[0], frame_size[1], 1))
    
    # Add the features to the frame buffer
    frame_buffer.append(features)
    
    # If the buffer is full, process the frames
    if len(frame_buffer) == buffer_size:
        # Convert the frame buffer to a numpy array
        frame_buffer_array = np.array(frame_buffer)
        
        # Reshape the array to (batch_size, sequence_length, features)
        frame_buffer_array = frame_buffer_array.reshape(1, buffer_size, -1)
        
        # Make predictions using the LSTM model
        predictions = model.predict(frame_buffer_array)
        
        # Get the predicted class label
        predicted_label = np.argmax(predictions)
        
        # Perform action based on the predicted label (e.g. alert nurse)
        if predicted_label == 1:
            print("Abnormal activity detected! Alerting nurse...")
        
        # Clear the frame buffer
        frame_buffer = []

    # Display the output
    cv2.imshow('TeleICU Monitoring System', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device
cap.release()
cv2.destroyAllWindows()