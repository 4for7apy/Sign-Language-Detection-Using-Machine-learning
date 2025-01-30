# import cv2
# import pickle
# import numpy as np
#
# # Load the trained ASL model
# with open('new_model.pickle', 'rb') as f:
#     model = pickle.load(f)
#
#
# # Function to preprocess the ROI (hand area) before prediction
# def preprocess_image(roi):
#     # Resize the ROI to 225x225
#     resized_image = cv2.resize(roi, (225, 225))
#
#     # Normalize pixel values to range [0, 1]
#     normalized_image = resized_image / 255.0
#
#     # Flatten the image to create a feature vector
#     flattened_image = normalized_image.flatten().reshape(1, -1)
#
#     return flattened_image
#
#
# # Map label indices to letters
# def get_predicted_label(label_index):
#     return chr(ord('a') + label_index)
#
#
# # Start the webcam for real-time gesture recognition
# def main():
#     print("Press 'q' to exit the webcam.")
#
#     # Open the webcam
#     cap = cv2.VideoCapture(0)
#
#     if not cap.isOpened():
#         print("Error: Could not access the webcam.")
#         return
#
#     while True:
#         # Capture a frame from the webcam
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to capture video.")
#             break
#
#         # Flip the frame horizontally (mirror effect)
#         frame = cv2.flip(frame, 1)
#
#         # Convert the frame to grayscale
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # Apply GaussianBlur to reduce noise
#         blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
#
#         # Apply thresholding to isolate the hand region
#         _, thresh_frame = cv2.threshold(blurred_frame, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#         # Find contours in the thresholded frame
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#         if contours:
#             # Find the largest contour (assuming it's the hand)
#             largest_contour = max(contours, key=cv2.contourArea)
#
#             # Get the bounding box for the largest contour
#             x, y, w, h = cv2.boundingRect(largest_contour)
#
#             # Draw a rectangle around the detected hand
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#             # Extract the ROI (Region of Interest) containing the hand
#             roi = gray_frame[y:y + h, x:x + w]
#
#             if roi.size > 0:  # Ensure ROI is valid
#                 # Preprocess the ROI
#                 processed_roi = preprocess_image(roi)
#
#                 # Predict the gesture
#                 predicted_label = model.predict(processed_roi)[0]
#                 predicted_gesture = get_predicted_label(predicted_label)
#
#                 # Display the prediction on the frame
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 cv2.putText(frame, f"Prediction: {predicted_gesture}", (10, 30), font, 1, (0, 255, 0), 2)
#
#         # Display the frames
#         cv2.imshow("ASL Gesture Recognition", frame)
#         cv2.imshow("Thresholded Frame", thresh_frame)
#
#         # Exit on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release the webcam and close all windows
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()


import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained ASL model
with open('new_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


# Function to preprocess the hand landmarks for prediction
def preprocess_landmarks(landmarks):
    # Flatten the landmarks into a 1D array
    flattened_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return flattened_landmarks.reshape(1, -1)


# Map label indices to letters
def get_predicted_label(label_index):
    return chr(ord('a') + label_index)


# Start the webcam for real-time gesture recognition
def main():
    print("Press 'q' to exit the webcam.")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (required by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        result = hands.process(rgb_frame)

        # Check if a hand is detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Preprocess the landmarks for the model
                landmarks = hand_landmarks.landmark
                processed_landmarks = preprocess_landmarks(landmarks)

                # Predict the gesture
                predicted_label = model.predict(processed_landmarks)[0]
                predicted_gesture = get_predicted_label(predicted_label)

                # Display the prediction on the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"Prediction: {predicted_gesture}", (10, 30), font, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("ASL Gesture Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
