import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Page configuration
st.set_page_config(page_title="Emotion Detection", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #ff4b2b, #ff416c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #333333;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">Emotion Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a face image to detect the emotion</div>', unsafe_allow_html=True)

# Rebuild CNN model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Load model weights
model.load_weights("final_emotion_model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion emoji mapping
emoji_map = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐"
}

# Upload section
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((48, 48))

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    prediction = model.predict(img_array)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    confidence = float(np.max(prediction)) * 100
    emoji = emoji_map.get(emotion, "")

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Stylish result box
    st.markdown(f"""
        <div class="result-box">
            {emoji} {emotion}
        </div>
    """, unsafe_allow_html=True)

    # Confidence display
    st.write(f"Confidence: {confidence:.2f}%")
    st.progress(int(confidence))
