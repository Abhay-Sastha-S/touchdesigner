import cv2
import mediapipe as mp
import numpy as np
import pygame
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Function to calculate distance between two landmarks
def calc_distance(lm1, lm2):
    return np.sqrt((lm1[0] - lm2[0]) ** 2 + (lm1[1] - lm2[1]) ** 2)

# Psychedelic shapes list
shapes = []

# Define hand-based properties
hand_properties = {
    'left': {'color': (255, 0, 0), 'size_multiplier': 2},
    'right': {'color': (0, 255, 0), 'size_multiplier': 3}
}

running = True
while running:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    screen.fill((0, 0, 0, 10))  # Create a fluid blending effect

    if result.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(result.multi_hand_landmarks):
            lm_list = [(lm.x * WIDTH, lm.y * HEIGHT) for lm in hand_landmarks.landmark]
            hand_label = 'left' if hand_no == 1 else 'right'

            # Extract key points
            index_tip = lm_list[8]
            thumb_tip = lm_list[4]
            wrist = lm_list[0]

            # Calculate features
            distance = calc_distance(index_tip, thumb_tip)
            rotation = np.arctan2(index_tip[1] - wrist[1], index_tip[0] - wrist[0])

            # Define manipulation properties based on hand and rotation
            size_multiplier = hand_properties[hand_label]['size_multiplier']
            base_color = hand_properties[hand_label]['color']
            color_variation = (int(rotation * 100) % 255, int(distance) % 255, 200)
            color = tuple((b + v) % 255 for b, v in zip(base_color, color_variation))
            size = int(distance * size_multiplier)
            angle = rotation * 100

            # Append new shape to list
            shapes.append({
                'pos': (int(index_tip[0]), int(index_tip[1])),
                'size': size,
                'color': color,
                'angle': angle
            })

            # Display measurements on hand tracking window
            cv2.putText(frame, f"Distance: {int(distance)}", (int(index_tip[0]), int(index_tip[1]) - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Rotation: {round(rotation, 2)}", (int(index_tip[0]), int(index_tip[1]) - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw psychedelic shapes
    for shape in shapes[-50:]:  # Keep last 50 shapes for smoother visuals
        pygame.draw.polygon(screen, shape['color'], [
            (shape['pos'][0] + shape['size'] * np.cos(shape['angle']), shape['pos'][1] + shape['size'] * np.sin(shape['angle'])),
            (shape['pos'][0] - shape['size'] * np.cos(shape['angle']), shape['pos'][1] - shape['size'] * np.sin(shape['angle'])),
            (shape['pos'][0] + shape['size'] * np.sin(shape['angle']), shape['pos'][1] - shape['size'] * np.cos(shape['angle'])),
            (shape['pos'][0] - shape['size'] * np.sin(shape['angle']), shape['pos'][1] + shape['size'] * np.cos(shape['angle']))
        ], 2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
