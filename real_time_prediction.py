import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from mediapipe_detection import mediapipe_detection
from draw_landmarks import draw_styled_landmarks
from keypoints_extraction import extract_keypoints

actions = np.array(['hello', 'thanks', 'iloveyou'])
try:
    model = load_model('action.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Initialize variables
sequence, sentence, predictions = [], [], []
threshold = 0.5

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open video capture.")
    exit(1)

with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Process frame and detect keypoints
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        
        # Extract keypoints and update sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Predict action if sequence length is sufficient
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            if res[np.argmax(res)] > threshold:
                sentence.append(actions[np.argmax(res)])
            if len(sentence) > 1:
                sentence = sentence[-1:]
        
        text = ''.join(sentence)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 5)
        text_width, text_height = text_size
        rect_x1, rect_y1 = 60, 60
        rect_x2, rect_y2 = rect_x1 + text_width + 20, rect_y1 + text_height + 20  # Added padding
        text_x = rect_x1 + 10  # Horizontal padding
        text_y = rect_y1 + text_height + 10
        # Display prediction
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 5)
        cv2.imshow('OpenCV Feed', frame)

        # Exit on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
