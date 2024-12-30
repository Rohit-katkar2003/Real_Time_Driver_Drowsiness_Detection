import streamlit as st
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pygame
import os 
import time

# Load the pre-trained CNN model for drowsiness detection
#model = load_model('drowsiness.h5')

# Define labels
labels = ['Closed', 'Open']

# Load face and eye cascade classifiers
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

# Function to initialize audio

def initialize_audio():
    try:
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("No display found. Skipping audio initialization.")
        pygame.mixer.init()
        pygame.mixer.music.load("alarm.wav")
    except Exception as e:
        print(f"Audio initialization failed: {e}")

initialize_audio()

# Initialize webcam
cam = cv.VideoCapture(0)
FRAME_WINDOW = st.image([])
run = st.button("Start")
stop = st.button("Stop")

# Function to preprocess the frame for model input
def preprocess_frame(frame):
    frame = cv.resize(frame, (24, 24))  # Resize to match model input shape
    frame = frame.astype("float") / 255.0  # Normalize
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    return frame

running = False
closed_start_time = None
alarm_triggered = False

if run:
    running = True

while running:
    ret, frame = cam.read()
    if not ret:
        st.warning("Failed to access webcam.")
        break
    
    frame = cv.flip(frame, 1)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    eye_status = 'Closed'
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        roi_gray = gray_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        # Draw rectangle around face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw ellipses around detected eyes
        if len(eyes) > 0:
            eye_status = 'Open'
            for (ex, ey, ew, eh) in eyes:
                center = (x + ex + ew // 2, y + ey + eh // 2)
                axes = (ew // 2, eh // 2)
                cv.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 0), 2)
            closed_start_time = None
            if alarm_triggered:
                pygame.mixer.music.stop()
                alarm_triggered = False
        else:
            if closed_start_time is None:
                closed_start_time = time.time()
            elif time.time() - closed_start_time >= 3:
                if not alarm_triggered:
                    try:
                        pygame.mixer.music.play()
                    except Exception as e:
                        print("Alarm triggered, but no audio device found.")
                    alarm_triggered = True

    # Display eye status text
    cv.putText(frame, f'Eyes: {eye_status}', (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    # Stop condition
    if stop:
        running = False
        cam.release()
        cv.destroyAllWindows()
        break

cam.release()
cv.destroyAllWindows()
