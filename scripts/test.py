import cv2
import joblib
import numpy as np
import mediapipe as mp


# Function to extract hand keypoints from an image
def extract_keypoints(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Flatten the 21 keypoints (x, y, z) into a single list
            keypoints = (
                [round(landmark.x, 2) for landmark in hand_landmarks.landmark]
                + [round(landmark.y, 2) for landmark in hand_landmarks.landmark]
                + [round(landmark.z, 2) for landmark in hand_landmarks.landmark]
            )
            return keypoints
        return None


# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the trained SVM model, labels and scaler
model = joblib.load("models/svm_hand_sign_model.pkl")
labels = joblib.load("models/labels.pkl")
scaler = joblib.load("models/scaler.pkl")

# Open webcam
cap = cv2.VideoCapture("video.mov")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # # Flip the frame horizontally for a mirrored effect
    # frame = cv2.flip(frame, 1)

    # Extract keypoints from the current frame
    keypoints = extract_keypoints(frame)

    if keypoints is not None:
        # Reshape keypoints into the format expected by the SVM model
        keypoints = np.array(keypoints).reshape(1, -1)

        # Standardize the keypoints (if you used StandardScaler during training)
        keypoints = scaler.transform(keypoints)

        # Make prediction using the trained SVM model
        prediction = model.predict(keypoints)

        # Display the predicted label
        label = prediction[0]
        cv2.putText(
            frame,
            f"Predicted Label: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    # Display the webcam feed with prediction
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
