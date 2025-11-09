import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Face Mesh with Iris
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

screen_w, screen_h = pyautogui.size()
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
    top = landmarks[eye_indices[1]]
    bottom = landmarks[eye_indices[5]]
    vert = np.linalg.norm(np.array([top.x * w, top.y * h]) -
                          np.array([bottom.x * w, bottom.y * h]))
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[3]]
    hori = np.linalg.norm(np.array([left.x * w, left.y * h]) -
                          np.array([right.x * w, right.y * h]))
    return vert / hori

cap = cv2.VideoCapture(0)

BLINK_THRESHOLD = 0.22
LONG_BLINK_TIME = 1.0
blink_start_time = None

# Calibration for gaze mapping
calibration_done = False
calib_left_iris = None
calib_right_iris = None

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

            # Iris landmarks: Left = 468, Right = 473
            left_iris = face_landmarks.landmark[468]
            right_iris = face_landmarks.landmark[473]

            # Get normalized iris positions (0-1)
            left_x, left_y = left_iris.x, left_iris.y
            right_x, right_y = right_iris.x, right_iris.y

            # Calibration: store center positions when looking straight
            if not calibration_done:
                calib_left_iris = (left_x, left_y)
                calib_right_iris = (right_x, right_y)
                calibration_done = True

            # Calculate relative gaze movement from calibration
            rel_left_x = left_x - calib_left_iris[0]
            rel_left_y = left_y - calib_left_iris[1]
            rel_right_x = right_x - calib_right_iris[0]
            rel_right_y = right_y - calib_right_iris[1]

            # Average both eyes' movement
            move_x = (rel_left_x + rel_right_x) / 2
            move_y = (rel_left_y + rel_right_y) / 2

            # Map to screen
            sensitivity = 1000  # adjust to control speed
            cursor_x = int(screen_w / 2 + move_x * sensitivity)
            cursor_y = int(screen_h / 2 + move_y * sensitivity)

            cursor_x = max(0, min(screen_w - 1, cursor_x))
            cursor_y = max(0, min(screen_h - 1, cursor_y))

            # Smooth cursor
            smooth_x, smooth_y = smooth_coords(cursor_x, cursor_y)
            pyautogui.moveTo(smooth_x, smooth_y)

            # Blink detection using EAR
            left_eye_indices = [33, 159, 158, 133, 153, 145]
            right_eye_indices = [362, 386, 387, 263, 373, 374]

            ear_left = eye_aspect_ratio(face_landmarks.landmark, left_eye_indices, w, h)
            ear_right = eye_aspect_ratio(face_landmarks.landmark, right_eye_indices, w, h)
            ear_avg = (ear_left + ear_right) / 2

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

            # Draw iris positions
            cv2.circle(frame, (int(left_x*w), int(left_y*h)), 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_x*w), int(right_y*h)), 3, (0, 255, 0), -1)

    cv2.imshow("Eye Controlled Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
