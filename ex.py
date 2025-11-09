import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
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

# EAR for blink detection
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand Tracking
    hand_results = hands.process(rgb_frame)
    h, w, _ = frame.shape
    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        # Index finger tip = landmark 8
        ix = int(hand_landmarks.landmark[8].x * screen_w)
        iy = int(hand_landmarks.landmark[8].y * screen_h)
        sx, sy = smooth_coords(ix, iy)
        pyautogui.moveTo(sx, sy)
        # Draw finger tip
        cv2.circle(frame, (int(hand_landmarks.landmark[8].x * w),
                           int(hand_landmarks.landmark[8].y * h)), 8, (0, 0, 255), -1)

    # Blink Detection
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
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
                    pyautogui.click()
                    print("Blink Detected 👀 -> Click 🖱️")
                blink_start_time = None

    cv2.imshow("Hand Cursor + Blink Click", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
 