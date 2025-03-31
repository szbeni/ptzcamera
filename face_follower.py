import cv2
import numpy as np
import requests
import time
import mediapipe as mp

# Camera settings
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# URL for controlling servos
YAW_URL = "http://10.1.1.26/angle0"
PITCH_URL = "http://10.1.1.26/angle1"

# PID Controller parameters
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
pid_yaw = PID(0.05, 0.00, 0.002)
pid_pitch = PID(0.06, 0.00, 0.0)

def send_duty_cycle(url, duty_cycle):
    requests.post(url, data={"angle": duty_cycle})

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.1)

def track_face():
    fps_last_time = 0
    fps = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                frame_height, frame_width, _ = frame.shape
                x, y, w, h = (int(bboxC.xmin * frame_width), 
                              int(bboxC.ymin * frame_height), 
                              int(bboxC.width * frame_width), 
                              int(bboxC.height * frame_height))
                cx, cy = x + w // 2, y + h // 2
                
                # Draw rectangle and center point
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                
                # Calculate error
                center_x, center_y = frame_width // 2, frame_height // 2
                error_x = center_x - cx
                error_y = center_y - cy
                
                # Compute PID outputs
                yaw_adjustment = pid_yaw.compute(error_x)
                pitch_adjustment = pid_pitch.compute(error_y)
                
                # Convert to duty cycle values
                yaw_duty = 90 + yaw_adjustment
                pitch_duty = 90 - pitch_adjustment
                
                yaw_duty = max(80, min(100, yaw_duty))
                pitch_duty = max(80, min(100, pitch_duty))
                
                send_duty_cycle(YAW_URL, yaw_duty)
                send_duty_cycle(PITCH_URL, pitch_duty)
                
                break  # Track only the first detected face


        else:
            yaw_duty = 90
            pitch_duty = 90


        # Display control demands on frame
        cv2.putText(frame, f"Yaw: {yaw_duty:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Pitch: {pitch_duty:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Calculate and display FPS
        current_time = time.time()
        frame_count += 1
        if current_time - fps_last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_last_time = current_time
        cv2.putText(frame, f"FPS: {fps}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        send_duty_cycle(YAW_URL, 90)
        send_duty_cycle(PITCH_URL, 90)



        # Show frame
        cv2.imshow("Face Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    send_duty_cycle(YAW_URL, 90)
    send_duty_cycle(PITCH_URL, 90)

if __name__ == "__main__":
    track_face()