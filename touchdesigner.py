import cv2
import mediapipe as mp
import numpy as np
import pygame
import math

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

def calc_distance(lm1, lm2):
    return np.sqrt((lm1[0] - lm2[0]) ** 2 + (lm1[1] - lm2[1]) ** 2)

# Mandala structure properties
structure = {
    'center': (WIDTH // 2, HEIGHT // 2),
    'scale': 100,
    'rotation': 0,
    'color': (255, 255, 255),
    'density': 6,
    'depth': 3
}

controls_text = """
Hand Tracking Controls:
- Index & Thumb distance: Controls Scale
- Wrist Rotation: Controls Rotation Speed
- Left Hand Distance: Controls Density
- Right Hand Distance: Controls Depth
- Both Hand Distance: Controls Color Variations
"""

def draw_mandala(center, scale, rotation, density, depth, color):
    if depth <= 0:
        return
    points = []
    for i in range(density):
        angle = rotation + (i * (2 * math.pi / density))
        x = center[0] + scale * math.cos(angle)
        y = center[1] + scale * math.sin(angle)
        points.append((x, y))
    pygame.draw.polygon(screen, color, points, 2)
    for p in points:
        draw_mandala(p, scale * 0.5, rotation + 0.2, density, depth - 1, color)

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
        hand_data = []
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = [(lm.x * WIDTH, lm.y * HEIGHT) for lm in hand_landmarks.landmark]
            index_tip = lm_list[8]
            thumb_tip = lm_list[4]
            wrist = lm_list[0]

            distance = calc_distance(index_tip, thumb_tip)
            rotation = np.arctan2(index_tip[1] - wrist[1], index_tip[0] - wrist[0])

            hand_data.append({'distance': distance, 'rotation': rotation})

            cv2.putText(frame, f"Distance: {int(distance)}", (int(index_tip[0]), int(index_tip[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Rotation: {round(rotation, 2)}", (int(index_tip[0]), int(index_tip[1]) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(hand_data) == 2:  # Combine both hands' inputs
            structure['scale'] = int((hand_data[0]['distance'] + hand_data[1]['distance']) / 2 * 2)
            structure['rotation'] += (hand_data[0]['rotation'] + hand_data[1]['rotation']) / 10
            structure['density'] = max(6, int(hand_data[0]['distance'] // 10) % 20)
            structure['depth'] = max(2, int(hand_data[1]['distance'] // 30) % 5)
            structure['color'] = (
                int((hand_data[0]['distance'] * 2) % 255),
                int((hand_data[1]['distance'] * 2) % 255),
                200
            )
    
    # Draw fractal mandala pattern
    draw_mandala(structure['center'], structure['scale'], structure['rotation'], structure['density'], structure['depth'], structure['color'])
    
    # Display control instructions
    font = pygame.font.Font(None, 24)
    lines = controls_text.split("\n")
    for i, line in enumerate(lines):
        text_surface = font.render(line, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10 + i * 20))
    
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
