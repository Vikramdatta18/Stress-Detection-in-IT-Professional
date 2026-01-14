import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

# Load trained model
model = load_model("face_stress_model.h5")

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess face
def preprocess_face(face):
    face = cv2.resize(face, (64, 64))  # Resize to match model input
    face = face / 255.0  # Normalize
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Extract face region
        face_resized = preprocess_face(face)  # Preprocess face

        # Make prediction
        prediction = model.predict(face_resized)[0][0]

        # **🔹 FIXED: FLIPPED LABELS 🔄**
        result = "Stressed" if prediction >= 0.5 else "Not Stressed"
        color = (0, 0, 255) if result == "Stressed" else (0, 255, 0)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw label above rectangle
        cv2.putText(frame, result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show webcam feed
    cv2.imshow("Live Stress Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
