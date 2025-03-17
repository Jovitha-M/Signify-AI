import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video file path (Change this to your actual video file)
video_path = "69541.mp4"  # Replace with your actual video filename

# Ensure the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    exit()

# Ask user for gesture label
gesture_label = input("Enter gesture label: ")  

# CSV file to store extracted data
csv_file = "asl_keypoints.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w") as f:
        f.write("frame," + ",".join([f"x{i},y{i},z{i}" for i in range(42)]) + ",label\n")  # 42 keypoints (21 per hand)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video file '{video_path}'.")
    exit()

frame_count = 0
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video has ended or cannot be read.")
        break  # Stop when video ends

    # Resize frame to speed up processing
    frame = cv2.resize(frame, (640, 480))

    # Convert to RGB (MediaPipe expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
    
    # Ensure we have 42 landmarks (2 hands * 21 keypoints)
    while len(keypoints) < 126:
        keypoints.append(0.0)  # Pad with zeros if missing keypoints

    if len(keypoints) == 126:
        keypoints.insert(0, frame_count)  # Add frame number at the beginning
        keypoints.append(gesture_label)  # Append label at the end
        data.append(keypoints)
        frame_count += 1

    # Show video with keypoints drawn
    cv2.imshow("ASL Gesture Extraction", frame)
    
    # Stop after collecting 30 frames for the gesture
    if frame_count >= 30:
        break

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(csv_file, mode="a", header=False, index=False)
print(f"Saved {frame_count} frames from video '{video_path}' for gesture '{gesture_label}'.")
