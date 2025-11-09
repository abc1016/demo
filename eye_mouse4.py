import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Camera
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# Blink detection variables
blink_threshold = 5.0
last_blink_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye landmarks
            landmarks = face_landmarks.landmark

            # Left eye points
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            left_eye_left = landmarks[33]
            left_eye_right = landmarks[133]

            # Eye aspect ratio for blink detection
            eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            eye_width = abs(left_eye_left.x - left_eye_right.x)
            ratio = eye_width / (eye_height + 1e-6)

            # Blink detection
            if ratio > blink_threshold and (time.time() - last_blink_time) > 1:
                pyautogui.click()
                last_blink_time = time.time()

            # Eye center
            eye_center_x = (left_eye_left.x + left_eye_right.x) / 2
            eye_center_y = (left_eye_top.y + left_eye_bottom.y) / 2

            # Convert to screen coordinates
            x = int(eye_center_x * screen_w)
            y = int(eye_center_y * screen_h)

            # Step-based control
            if eye_center_x < 0.4:  # Look left
                pyautogui.moveRel(-50, 0, duration=0.1)
            elif eye_center_x > 0.6:  # Look right
                pyautogui.moveRel(50, 0, duration=0.1)
            if eye_center_y < 0.4:  # Look up
                pyautogui.moveRel(0, -50, duration=0.1)
            elif eye_center_y > 0.6:  # Look down
                pyautogui.moveRel(0, 50, duration=0.1)

    cv2.imshow("Eye Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

