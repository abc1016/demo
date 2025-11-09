import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# To smooth the movement
history = []

def smooth_coords(x, y, history_len=5):
    history.append((x, y))
    if len(history) > history_len:
        history.pop(0)
    avg_x = int(np.mean([p[0] for p in history]))
    avg_y = int(np.mean([p[1] for p in history]))
    return avg_x, avg_y

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame so movement feels natural
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Left eye landmarks (example points: 33, 133)
            eye_x = (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2
            eye_y = (face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2

            # Map to screen size
            cursor_x = int(eye_x * screen_w)
            cursor_y = int(eye_y * screen_h)

            # Smooth the movement
            smooth_x, smooth_y = smooth_coords(cursor_x, cursor_y)

            # Move cursor
            pyautogui.moveTo(smooth_x, smooth_y)

            # Show eye point on webcam
            h, w, _ = frame.shape
            cx, cy = int(eye_x * w), int(eye_y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("Eye Controlled Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
