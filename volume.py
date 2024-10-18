import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)  # Adjust confidence
mp_drawing = mp.solutions.drawing_utils

# Initialize volume control parameters
volume_step = 2  # Increased volume change step for slower changes
current_volume = 50  # Initial volume

# Open webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Extract hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand landmarks for specific points (wrist, thumb, and index finger)
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Check thumb and index finger vertical positions for sensitivity
            thumb_vertical_pos = thumb.y * frame.shape[0]  # Normalize to frame height
            index_finger_vertical_pos = index_finger.y * frame.shape[0]  # Normalize to frame height

            # Check if thumb is above the index finger (volume up gesture)
            if thumb_vertical_pos < index_finger_vertical_pos - 30:  # Adjust threshold for sensitivity
                current_volume = min(100, current_volume + volume_step)  # Increase volume
            # Check if thumb is below the index finger (volume down gesture)
            elif thumb_vertical_pos > index_finger_vertical_pos + 30:  # Adjust threshold for sensitivity
                current_volume = max(0, current_volume - volume_step)  # Decrease volume

    # Display current volume on the frame
    cv2.putText(frame, f"Volume: {current_volume}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Virtual volume', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
