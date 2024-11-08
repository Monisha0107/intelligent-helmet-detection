import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained helmet detection model
model = load_model('models/helmet-1.h5')

# Function to perform helmet detection on a frame
def detect_helmet(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = img_to_array(resized_frame) / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0)

    # Predict helmet status
    prediction = model.predict(input_tensor)[0][0]
    label = "Helmet" if prediction > 0.5 else "No Helmet"

    print(f"Prediction Score: {prediction}")

    color = (0, 255, 0) if label == "Helmet" else (0, 0, 255)
    cv2.rectangle(frame, (50, 50), (frame.shape[1] - 50, frame.shape[0] - 50), color, 2)
    cv2.putText(frame, f"{label}: {prediction:.2f}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame, label

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform helmet detection
    frame, label = detect_helmet(frame)
    print(f"Detection Result: {label}")

    # Display the resulting frame with detection results
    cv2.imshow('Helmet Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



