import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # Track two hands (left and right)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Create a white canvas (background) for drawing (fullscreen)
canvas = np.ones((1080, 1920, 3), dtype=np.uint8) * 255  # White canvas

# Define colors (6 colors in the color wheel)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 165, 0)]
current_color = colors[0]  # Start with Red

# Color wheel for visualization (simple static wheel)
color_wheel_radius = 100  # Radius of the color wheel
wheel_center = (1600, 100)  # Position on the canvas

# Open webcam
cap = cv2.VideoCapture(0)

# To store the previous position of the index finger
prev_x, prev_y = -1, -1

# To track the drawing/eraser mode
eraser_mode = False
eraser_radius = 30  # Size of the eraser (radius of the circle)

# Initialize rotation angle for left hand color control
prev_angle = None

# Create a fullscreen window for the canvas
cv2.namedWindow("Drawing Canvas", cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty("Drawing Canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert to RGB for MediaPipe
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Draw landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of the index finger tip (landmark 8) and middle finger tip (landmark 12)
            index_x = int(hand_landmarks.landmark[8].x * width)
            index_y = int(hand_landmarks.landmark[8].y * height)

            middle_x = int(hand_landmarks.landmark[12].x * width)
            middle_y = int(hand_landmarks.landmark[12].y * height)

            # Get the current hand (left or right)
            if hand_label.classification[0].label == 'Left':
                # Check if the left hand is rotating for color selection
                left_thumb_x = int(hand_landmarks.landmark[4].x * width)
                left_thumb_y = int(hand_landmarks.landmark[4].y * height)

                # Calculate the angle between the thumb and index finger (landmark 4 and 8)
                dx = left_thumb_x - index_x
                dy = left_thumb_y - index_y
                angle = math.atan2(dy, dx) * 180 / math.pi

                if prev_angle is not None:
                    # Check the difference in angle to decide if we have made a significant rotation
                    angle_diff = angle - prev_angle
                    if abs(angle_diff) > 30:  # Threshold for rotation
                        color_index = int((angle + 180) // 60) % len(colors)
                        current_color = colors[color_index]
                prev_angle = angle

            # Right hand: Drawing and erasing
            if hand_label.classification[0].label == 'Right':
                # Check if two fingers (index and middle) are close enough to activate eraser mode
                distance = np.sqrt((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2)
                if distance < 50:  # Adjust this threshold as needed
                    eraser_mode = True
                    cv2.putText(frame, "Eraser Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    eraser_mode = False
                    cv2.putText(frame, "Drawing Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw or erase based on the mode
                if prev_x != -1 and prev_y != -1:
                    if eraser_mode:
                        # Erase by drawing black (same color as background) over the previous path
                        cv2.circle(canvas, (index_x, index_y), eraser_radius, (255, 255, 255), -1)  # White color to erase
                    else:
                        # Draw on the canvas with the current color
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), current_color, 10)

                # Update the previous coordinates for the next frame
                prev_x, prev_y = index_x, index_y

            # Draw the current color box
            color_box_start = (10, 100)
            color_box_end = (200, 200)
            cv2.rectangle(canvas, color_box_start, color_box_end, current_color, -1)

            # Draw a text label to show which color we are on
            color_label = f"Current Color: {current_color}"
            cv2.putText(canvas, color_label, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Draw color wheel
            color_wheel_img = np.zeros((color_wheel_radius * 2, color_wheel_radius * 2, 3), dtype=np.uint8)
            cv2.circle(color_wheel_img, (color_wheel_radius, color_wheel_radius), color_wheel_radius, current_color, -1)
            canvas[100:100 + color_wheel_img.shape[0], 1400:1400 + color_wheel_img.shape[1]] = color_wheel_img

    # Show the canvas in a fullscreen window
    cv2.imshow("Drawing Canvas", canvas)

    # Show the camera feed with the mode and current color
    cv2.imshow("Camera Feed", frame)

    # Quit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
