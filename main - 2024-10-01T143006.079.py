import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Use Face Detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Convert frame color
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face
        results = face_detection.process(frame_rgb)

        # If face is detected
        if results.detections:
            for detection in results.detections:
                # Draw detection on the frame
                mp_drawing.draw_detection(frame, detection)
                
                # Extract bounding box of the face
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * frame.shape[1]), int(bboxC.ymin * frame.shape[0]), \
                             int(bboxC.width * frame.shape[1]), int(bboxC.height * frame.shape[0])
                
                # Calculate center of face (head movement proxy)
                center_x, center_y = x + w // 2, y + h // 2

                # Move camera or trigger action based on head position
                if center_x < frame.shape[1] // 3:
                    print("Move camera left")
                elif center_x > 2 * frame.shape[1] // 3:
                    print("Move camera right")
                else:
                    print("Keep camera center")

        # Display frame
        cv2.imshow('Camera', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
