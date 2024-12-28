import cv2 as cv
import numpy as np
import time
import pygame
import streamlit as st
from PIL import Image

classes = ['Closed', 'Open']  # Class labels for eyes only

# Load face and eye cascade classifiers
face_cascade = cv.CascadeClassifier("haarcascade files/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascade files/haarcascade_eye.xml")

# Initialize alarm
pygame.mixer.init()
pygame.mixer.music.load("research/alarm.wav")

# Track eye closure duration
closed_start_time = None
alarm_triggered = False

# Prepare the frame for eye detection
def prepare_frame(frame):
    global closed_start_time, alarm_triggered
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert to grayscale for better detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    eye_status = 'Closed'
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        roi_gray = gray_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        # If two or more eyes detected, classify as 'Open'
        if len(eyes) >= 2:
            eye_status = 'Open'
            if alarm_triggered:
                pygame.mixer.music.stop()
                alarm_triggered = False
            closed_start_time = None
        else:
            if closed_start_time is None:
                closed_start_time = time.time()
            elif time.time() - closed_start_time >= 3:
                if not alarm_triggered:
                    pygame.mixer.music.play()
                    alarm_triggered = True
        
        # Draw a rectangle around the entire face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        break  # Stop after detecting the first face to avoid multiple rectangles

    cv.putText(frame, eye_status, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, eye_status

# Streamlit interface
st.title("Real-time Eye Detection and Drowsiness Alert")
st.markdown("**Close your eyes for 3 seconds to trigger the alarm.**")

FRAME_WINDOW = st.image([])
run = st.button("Start Detection")
stop = st.button("Stop Detection")

cap = cv.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to access webcam.")
        break

    frame = cv.flip(frame, 1)  # Flip frame horizontally
    frame, eye_status = prepare_frame(frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    if stop:
        cap.release()
        cv.destroyAllWindows()
        break

cap.release()
cv.destroyAllWindows()
