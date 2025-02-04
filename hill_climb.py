import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # Allow for detection of both hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Open the webcam.
cap = cv2.VideoCapture(0)

# Variables for calibration and cooldown.
calibrated_left = False
calibrated_right = False
center_left_x, center_left_y = 0, 0
center_right_x, center_right_y = 0, 0
last_command = None  # To store the last command sent
last_command_time = time.time()
command_delay = 0.1  # Minimum time between commands (in seconds)

# To track the last fist state
last_fist_state = None

# Function to check if fist is open or closed based on thumb and index finger distance
def is_fist_closed(hand_landmarks):
    # Get thumb and index finger tip positions
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    # Calculate the Euclidean distance between the thumb and index finger tips
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    
    # If the distance is less than a certain threshold, it's a closed fist
    threshold = 0.1  # Adjust this threshold based on your use case
    return distance < threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror view.
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert the frame to RGB for MediaPipe.
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks.
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine if the current hand is left or right
            if hand_label.classification[0].label == 'Left':
                if not calibrated_left:
                    center_left_x, center_left_y = hand_landmarks.landmark[8].x * width, hand_landmarks.landmark[8].y * height
                    calibrated_left = True
                    cv2.putText(frame, "Left Hand Calibrated", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                if not calibrated_right:
                    center_right_x, center_right_y = hand_landmarks.landmark[8].x * width, hand_landmarks.landmark[8].y * height
                    calibrated_right = True
                    cv2.putText(frame, "Right Hand Calibrated", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check if fist is open or closed.
            fist_closed = is_fist_closed(hand_landmarks)

            # Check if there is a change in fist state
            if fist_closed != last_fist_state:
                # If fist is closed, press 'down', if open, press 'up'
                if fist_closed:
                    print("Fist closed - pressing down")
                    pyautogui.keyDown('down')
                    pyautogui.keyUp('up')  # Make sure 'up' key is released
                else:
                    print("Fist open - pressing up")
                    pyautogui.keyDown('up')
                    pyautogui.keyUp('down')  # Make sure 'down' key is released

                last_fist_state = fist_closed  # Update fist state

    # Display the frame.
    cv2.imshow("Hand Gesture Control - Fist", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up.
cap.release()
cv2.destroyAllWindows()
