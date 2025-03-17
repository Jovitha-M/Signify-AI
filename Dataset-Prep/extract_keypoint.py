import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


video_path = "69541.mp4"  


if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    exit()


gesture_label = input("Enter gesture label: ")  


csv_file = "asl_keypoints.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w") as f:
        f.write("frame," + ",".join([f"x{i},y{i},z{i}" for i in range(42)]) + ",label\n")  # 42 keypoints (21 per hand)


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file '{video_path}'.")
    exit()

frame_count = 0
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video has ended or cannot be read.")
        break 
  
    frame = cv2.resize(frame, (640, 480))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
    
    while len(keypoints) < 126:
        keypoints.append(0.0) 

    if len(keypoints) == 126:
        keypoints.insert(0, frame_count)  
        keypoints.append(gesture_label)  
        data.append(keypoints)
        frame_count += 1

   
    cv2.imshow("ASL Gesture Extraction", frame)
    
   
    if frame_count >= 30:
        break

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv(csv_file, mode="a", header=False, index=False)
print(f"Saved {frame_count} frames from video '{video_path}' for gesture '{gesture_label}'.")
