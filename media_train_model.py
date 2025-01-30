import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Dataset folder path
dataset_folder = "D:/asl project/ASL Interpration/Sign-Language-to-Text-and-Speech/New_DataSet"

# Prepare dataset arrays
X = []  # Feature vectors (landmarks)
y = []  # Labels

# Iterate through gesture folders (e.g., "A", "B", "C", etc.)
for gesture in os.listdir(dataset_folder):
    gesture_path = os.path.join(dataset_folder, gesture)

    if os.path.isdir(gesture_path):
        print(f"Processing gesture: {gesture}")

        for img_file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_file)

            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read image: {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe Hands
            result = hands.process(image)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Extract and flatten landmarks
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    X.append(landmarks)
                    y.append(gesture)  # Use folder name as the label
            else:
                print(f"No hand detected in image: {img_path}")

# Check if X and y have valid data
if len(X) == 0 or len(y) == 0:
    print("No data found! Check your dataset folder or image processing.")
    exit()

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print(f"Dataset created with {len(X)} samples and {len(X[0])} features per sample.")

# Train the model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open("landmark_model.pickle", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
