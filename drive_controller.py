import cv2
import mediapipe as mp 
import numpy as np
import time

cap = cv2.VideoCapture(1)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

left_eye = [33, 160, 158, 133, 153, 144]
right_eye = [362, 385, 387, 263, 373, 380]

eyes_closed_start_time = None
eyes_closed_duration = 0.0
CLOSED_THRESHOLD = 2.0

def eye_aspect_ratio(eye):
    eye = np.array(eye)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            h, w, c = frame.shape
            
            left_eye_points = []
            for idx in left_eye:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye_points.append((x, y))
            
            right_eye_points = []
            for idx in right_eye:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye_points.append((x, y))
            
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            ear = (left_ear + right_ear) / 2.0
            
            if ear < 0.25:
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = time.time()
                else:
                    eyes_closed_duration = time.time() - eyes_closed_start_time
                
                if eyes_closed_duration >= CLOSED_THRESHOLD:
                    cv2.putText(frame, "Car is stopped", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Eyes: CLOSED", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Closed for: {eyes_closed_duration:.1f}s", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "WARNING: EYES CLOSED TOO LONG!", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "Car is moving forward", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                    cv2.putText(frame, "Eyes: Blinking...", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    cv2.putText(frame, f"Closed for: {eyes_closed_duration:.1f}s", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            else:
                eyes_closed_start_time = None
                eyes_closed_duration = 0.0
                
                cv2.putText(frame, "Car is moving forward", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Eyes: OPEN", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    else:
        eyes_closed_start_time = None
        eyes_closed_duration = 0.0
        cv2.putText(frame, "No face detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()