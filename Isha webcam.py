import cv2 as cv
import mediapipe as mp
import time

# Initialize MediaPipe solutions for pose and hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the camera
cap = cv.VideoCapture(0)

# Set camera resolution
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  # Set width to 1280 pixels
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)  # Set height to 720 pixels

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# FPS calculation variables
fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror view
    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame for pose and hand landmarks
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    # Draw body landmarks
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Draw hand landmarks
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )

            # Highlight finger tips and joints
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)

                # Differentiate fingertips from other joints
                if idx in [4, 8, 12, 16, 20]:  # Fingertip landmarks
                    cv.circle(frame, (x, y), 8, (0, 255, 0), cv.FILLED)
                else:
                    cv.circle(frame, (x, y), 5, (255, 0, 0), cv.FILLED)

    # Calculate FPS
    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = time.time()
        frame_count = 0

    # Display FPS on the frame
    cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # Resize the frame to make it larger
    frame = cv.resize(frame, (1280, 720))

    # Show the output frame
    cv.imshow("Advanced Pose and Finger Tracking", frame)

    # Resize the display window
    cv.resizeWindow("Advanced Pose and Finger Tracking", 1280, 720)

    # Exit when 'q' is pressed
    key = cv.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
