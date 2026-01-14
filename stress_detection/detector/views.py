from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'face_stress_model1.h5')  # Ensure the path is correct
model = load_model(model_path)

# Load OpenCV Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Home page view
def home(request):
    return render(request, 'home.html')

# Function to process video frames and perform stress detection
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # Extract face region
            face_resized = cv2.resize(face, (64, 64)) / 255.0  # Resize & normalize
            face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension

            # Predict stress level
            prediction = model.predict(face_resized)[0][0]
            label = "Stressed" if prediction >= 0.5 else "Not Stressed"
            color = (0, 0, 255) if label == "Stressed" else (0, 255, 0)  # Red for stressed, Green for not stressed

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display label above rectangle
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# View to serve the live video feed
def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
