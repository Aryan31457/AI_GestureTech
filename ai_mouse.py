import cv2
import mediapipe as mp
import pyautogui

# Initialize video capture, hand detection, and drawing utilities
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Reduce frame size for better performance
frame_reduction = 2

index_y = 0  # To track the y position of the index finger

while True:
    success, frame = cap.read()
    if not success:
        break

    # Reduce frame size for faster processing (optional)
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    reduced_frame = cv2.resize(frame, (frame_width // frame_reduction, frame_height // frame_reduction))

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(reduced_frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            # Draw landmarks on the reduced frame
            drawing_utils.draw_landmarks(reduced_frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * (frame_width // frame_reduction))
                y = int(landmark.y * (frame_height // frame_reduction))

                # Detect index finger (landmark id 8)
                if id == 8:
                    cv2.circle(img=reduced_frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    index_x = screen_width / (frame_width // frame_reduction) * x
                    index_y = screen_height / (frame_height // frame_reduction) * y

                # Detect thumb (landmark id 4)
                if id == 4:
                    cv2.circle(img=reduced_frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    thumb_x = screen_width / (frame_width // frame_reduction) * x
                    thumb_y = screen_height / (frame_height // frame_reduction) * y

                    # Check the distance between thumb and index finger for click detection
                    if abs(index_y - thumb_y) < 30:
                        pyautogui.click()
                        pyautogui.sleep(0.25)  # Reduce the sleep time for less delay
                    elif abs(index_y - thumb_y) < 100:
                        pyautogui.moveTo(index_x, index_y, duration=0.1)  # Smoother and faster movement

    # Display the reduced frame with hand landmarks
    cv2.imshow('Virtual Mouse', reduced_frame)

    # Check if 'q' key is pressed to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
