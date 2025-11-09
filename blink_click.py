import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices (from MediaPipe face mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # left eye keypoints
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # right eye keypoints

def eye_aspect_ratio(eye_points, landmarks, frame_w, frame_h):
    # Convert normalized landmarks to pixel coordinates
    coords = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_points]

    # vertical distances
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    # horizontal distance
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))

    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)
blink_counter = 0
blink_threshold = 0.21  # lower = stricter blink detection
closed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Draw face mesh
            mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # EAR for left & right eye
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks.landmark, w, h)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks.landmark, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < blink_threshold:
                closed_frames += 1
            else:
                if closed_frames > 2:  # eye was closed for a few frames
                    blink_counter += 1
                    print("Blink Detected 👀 -> Click 🖱️")
                    pyautogui.click()   # perform mouse click
                closed_frames = 0

    cv2.imshow("Blink to Click", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
