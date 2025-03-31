import cv2
import numpy as np
import requests
import time
import mediapipe as mp
import pyvirtualcam

# Camera settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# URL for controlling servos
YAW_URL = "http://10.1.1.26/angle0"
PITCH_URL = "http://10.1.1.26/angle1"

# PID Controller class
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        self.prev_error = error
        self.last_time = current_time
        
        return output

# Initialize PID controllers
pid_yaw = PID(0.037, 0.00, 0.005)
pid_pitch = PID(0.045, 0.00, 0.0)

def send_duty_cycle(url, duty_cycle):
    requests.post(url, data={"angle": duty_cycle})

# Initialize face detection options
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.2)

def detect_hands_mediapipe(frame):
    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # List of hand landmarks for each hand detected
        hand_landmarks = results.multi_hand_landmarks
        hand_bboxes = []
        
        # Get bounding box around each hand
        for landmarks in hand_landmarks:
            h, w, _ = frame.shape
            x_min = min([landmark.x for landmark in landmarks.landmark])
            y_min = min([landmark.y for landmark in landmarks.landmark])
            x_max = max([landmark.x for landmark in landmarks.landmark])
            y_max = max([landmark.y for landmark in landmarks.landmark])
            
            # Convert relative coordinates to absolute pixel values
            x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
            hand_bboxes.append((x_min, y_min, x_max - x_min, y_max - y_min))  # (x, y, width, height)
        
        return hand_bboxes
    
def detect_face_mediapipe(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
        return (x, y, w, h)
    return None

def detect_face_opencv(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return (x, y, w, h)
    return None

def detect_red_dot(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define the range for detecting red in HSV
    # Red color has two ranges in HSV space
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks
    red_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        #print(cv2.contourArea(c))
        if cv2.contourArea(c) < 200:
            return None
        x, y, w, h = cv2.boundingRect(c)
        return (x, y, w, h)
    
    return None

def track_object():
    modes = ["mediapipe", "opencv", "red_dot", "hands"]
    mode_index = 0
    last_mode_switch = time.time()
    fps_last_time = time.time()
    frame_count = 0
    fps = 0
    
    with pyvirtualcam.Camera(width=640, height=480, fps=30) as virtcam:
        while True:
            ret, frame = cap.read()
            frame_orig = frame.copy()
            virtcam.send(frame_orig)

            if not ret:
                break
            
            key = cv2.waitKey(1)
            if  key & 0xFF == ord('m'):
                mode_index = (mode_index + 1) % len(modes)
                #last_mode_switch = time.time()
            elif key & 0xFF == ord('q'):
                break

            mode = modes[mode_index]
            
            if mode == "mediapipe":
                bbox = detect_face_mediapipe(frame)
            elif mode == "opencv":
                bbox = detect_face_opencv(frame)
            elif mode == "red_dot":
                bbox = detect_red_dot(frame)
            elif mode == "hands":
                bbox = detect_hands_mediapipe(frame)
            

            if bbox:
                if isinstance(bbox, list):
                    bbox = bbox[0]

                x, y, w, h = bbox
                cx, cy = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                
                frame_width, frame_height = frame.shape[1], frame.shape[0]
                center_x, center_y = frame_width // 2, frame_height // 2
                error_x = center_x - cx
                error_y = center_y - cy
                
                yaw_adjustment = pid_yaw.compute(error_x)
                pitch_adjustment = pid_pitch.compute(error_y)
                
                yaw_duty = max(80, min(100, 90 + yaw_adjustment))
                pitch_duty = max(80, min(100, 90 - pitch_adjustment))
            else:
                yaw_duty = 90
                pitch_duty = 90
            
            frame_count += 1
            if time.time() - fps_last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_last_time = time.time()
            
            send_duty_cycle(YAW_URL, yaw_duty)
            send_duty_cycle(PITCH_URL, pitch_duty)

            
            cv2.putText(frame, f"Yaw: {yaw_duty:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch_duty:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Mode: {mode}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Tracking", frame)
        

    
    cap.release()
    cv2.destroyAllWindows()
    #send_duty_cycle(YAW_URL, 90)
    #send_duty_cycle(PITCH_URL, 90)

if __name__ == "__main__":
    track_object()
