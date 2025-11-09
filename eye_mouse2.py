import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# For smoothing cursor movement
history = []

def smooth_coords(x, y, history_len=5):
    history.append((x, y))
    if len(history) > history_len:
        history.pop(0)
    avg_x = int(np.mean([p[0] for p in history]))
    avg_y = int(np.mean([p[1] for p in history]))
    return avg_x, avg_y

# Eye Aspect Ratio (EAR) for blink detection
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    # Vertical distances
    top = landmarks[eye_indices[1]]
    bottom = landmarks[eye_indices[5]]
    vert = np.linalg.norm(np.array([top.x * w, top.y * h]) -
                          np.array([bottom.x * w, bottom.y * h]))
    # Horizontal distance
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[3]]
    hori = np.linalg.norm(np.array([left.x * w, left.y * h]) -
                          np.array([right.x * w, right.y * h]))
    return vert / hori

cap = cv2.VideoCapture(0)

# Blink parameters
BLINK_THRESHOLD = 0.22   # Lower = more sensitive
LONG_BLINK_TIME = 1.0    # Seconds for right click
blink_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Left eye center
            left_eye_x = (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2
            left_eye_y = (face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2

            # Right eye center
            right_eye_x = (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2
            right_eye_y = (face_landmarks.landmark[362].y + face_landmarks.landmark[263].y) / 2

            # Average of both eyes
            eye_x = (left_eye_x + right_eye_x) / 2
            eye_y = (left_eye_y + right_eye_y) / 2

            # Map to screen size
            cursor_x = int(eye_x * screen_w)
            cursor_y = int(eye_y * screen_h)

            # Smooth cursor
            smooth_x, smooth_y = smooth_coords(cursor_x, cursor_y)
            pyautogui.moveTo(smooth_x, smooth_y)

            # Draw tracking point
            cx, cy = int(eye_x * w), int(eye_y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # EAR for blink detection
            left_indices = [33, 159, 158, 133, 153, 145]
            right_indices = [362, 386, 387, 263, 373, 374]

            ear_left = eye_aspect_ratio(face_landmarks.landmark, left_indices, w, h)
            ear_right = eye_aspect_ratio(face_landmarks.landmark, right_indices, w, h)
            ear_avg = (ear_left + ear_right) / 2

            # Blink detection
            if ear_avg < BLINK_THRESHOLD:
                if blink_start_time is None:
                    blink_start_time = time.time()
            else:
                if blink_start_time is not None:
                    blink_duration = time.time() - blink_start_time
                    if blink_duration < LONG_BLINK_TIME:
                        pyautogui.click()
                        print("Short Blink 👀 -> Left Click 🖱️")
                    else:
                        pyautogui.click(button='right')
                        print("Long Blink 👀 -> Right Click 🖱️")
                blink_start_time = None

    cv2.imshow("Eye Controlled Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
