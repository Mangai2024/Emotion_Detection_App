Emotion Detection using CNN
Project Overview

This project is a deep learning–based web application that detects human emotions from facial images.
It uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset and is deployed as an interactive web app using Streamlit on Hugging Face Spaces.

The model classifies images into seven emotion categories.

Emotions Detected

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

Tech Stack

Python

TensorFlow / Keras

NumPy

Pillow

Streamlit

Hugging Face Spaces

Project Workflow

Data collection using FER-2013 dataset

Image preprocessing and normalization

Building a CNN model

Training and evaluating the model

Saving trained model weights

Creating a Streamlit web interface

Deploying the app on Hugging Face

Model Architecture

The CNN model consists of:

Convolutional layers for feature extraction

MaxPooling layers for dimensionality reduction

Fully connected dense layers

Dropout for regularization

Softmax output layer for emotion classification

How to Run Locally

Clone the repository:

git clone https://github.com/your-username/emotion-detection.git


Navigate to the project folder:

cd emotion-detection


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py

Live Demo

Deployed on Hugging Face:
https://huggingface.co/spaces/Mangai2024/Emotion_Detection

Features

Upload a face image

Predict emotion instantly

Confidence score display

Clean and interactive UI

Challenges Faced

Model file path issues during deployment

TensorFlow and Python version conflicts

Keras model deserialization errors

Solved by rebuilding architecture and loading weights

Future Improvements

Real-time webcam emotion detection

Multi-face detection

Model accuracy improvement

Mobile-friendly UI

Author

Mangai
Aspiring Data Scientist / AI Engineer
