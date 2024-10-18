import cv2
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller, Key

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the keyboard controller
keyboard = Controller()

# Define the virtual keyboard layout with all alphabets and some special characters
keyboard_layout = [
    ['Esc', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12'],
    ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'Backspace'],
    ['Tab', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
    ['CapsLock', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'", 'Enter'],
    ['shift_l', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'shift_r'],
    ['Ctrl', 'Win', 'Alt', 'Space', 'Alt', 'Win', 'Menu', 'Ctrl']
]

# Open webcam
cap = cv2.VideoCapture(0)
pointer_radius = 15
clicked = False  # Flag to track click state

# Set the desired frame dimensions
frame_width = 800  # Width of the frame
frame_height = 600  # Height of the frame
frame_margin = 20  # Margin from the edges
key_spacing = 5  # Spacing between keys

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Increase frame width and add margins
    frame = cv2.resize(frame, (frame_width, frame_height))
    frame = cv2.copyMakeBorder(frame, frame_margin, frame_margin, frame_margin, frame_margin,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))  # Add black margins

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the position of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x = int(index_finger_tip.x * frame.shape[1])
            index_finger_y = int(index_finger_tip.y * frame.shape[0])

            # Check if the index finger tip is within the keyboard area
            if frame_margin < index_finger_x < frame_width - frame_margin and \
                    frame_margin < index_finger_y < frame_height - frame_margin:
                row_index = (index_finger_y - frame_margin) // ((frame_height - 2 * frame_margin) // len(keyboard_layout))
                col_index = (index_finger_x - frame_margin) // ((frame_width - 2 * frame_margin) // len(keyboard_layout[0]))
                if 0 <= row_index < len(keyboard_layout) and 0 <= col_index < len(keyboard_layout[row_index]):
                    key_pressed = keyboard_layout[row_index][col_index]
                    cv2.circle(frame, (index_finger_x, index_finger_y), pointer_radius, (0, 255, 0), cv2.FILLED)

            # Check for a click gesture (thumb and index finger close together)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x = int(thumb_tip.x * frame.shape[1])
            thumb_y = int(thumb_tip.y * frame.shape[0])
            thumb_distance = np.sqrt((index_finger_x - thumb_x)**2 + (index_finger_y - thumb_y)**2)
            if thumb_distance < 30 and not clicked:  # Adjust the threshold for click detection
                key_pressed = keyboard_layout[row_index][col_index]
                keyboard.press(key_pressed)
                clicked = True  # Set clicked flag
            elif thumb_distance >= 30 and clicked:
                keyboard.release(key_pressed)
                clicked = False  # Reset clicked flag

    # Display the virtual keyboard
    key_width = (frame_width - 2 * frame_margin - (len(keyboard_layout[0]) - 1) * key_spacing) // len(keyboard_layout[0])
    key_height = (frame_height - 2 * frame_margin - (len(keyboard_layout) - 1) * key_spacing) // len(keyboard_layout)
    for i, row in enumerate(keyboard_layout):
        for j, key in enumerate(row):
            key_x = frame_margin + j * (key_width + key_spacing)
            key_y = frame_margin + i * (key_height + key_spacing)
            cv2.rectangle(frame, (key_x, key_y), (key_x + key_width, key_y + key_height), (255, 255, 255), 2)
            cv2.putText(frame, key, (key_x + key_width//3, key_y + key_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Hand-controlled Keyboard', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
