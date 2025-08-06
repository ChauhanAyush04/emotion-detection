import streamlit as st
from streamlit_webrtc import  webrtc_streamer, VideoProcessorBase
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import av

# Load trained model
model = load_model("emotion_model.h5", compile=False)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Page setup
st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Facial Emotion Recognition</h1>", unsafe_allow_html=True)
st.markdown("Upload an image or use your webcam to detect emotions.")

# ------------------ Webcam Detection ------------------ #
class EmotionDetector(VideoProcessorBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            emotion = class_names[np.argmax(preds)]
            confidence = np.max(preds)

            label = f"{emotion} ({confidence:.0%})"
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Show webcam section
st.markdown("### 📷 Use Webcam:")
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False}
)

# ------------------ Image Upload Section ------------------ #
st.markdown("---")
st.markdown("### 📁 Or Upload an Image:")

uploaded_file = st.file_uploader("Upload an image with one or more faces", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected in the uploaded image.")
    else:
        for i, (x, y, w, h) in enumerate(faces):
            roi_gray = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi_gray, (48, 48)) / 255.0
            roi = roi.reshape(1, 48, 48, 1)

            prediction = model.predict(roi, verbose=0)[0]
            emotion_idx = np.argmax(prediction)
            emotion = class_names[emotion_idx]
            confidence = prediction[emotion_idx]

            # Draw rectangle & label on image
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion} ({confidence:.0%})"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

            # Display result + chart
            st.markdown(f"#### Face {i+1}: {emotion} ({confidence:.2f})")
            st.bar_chart({class_names[j]: float(prediction[j]) for j in range(len(class_names))})

        # Show the image with all face boxes
        st.image(image, channels="BGR", caption="Detected Faces and Emotions")
