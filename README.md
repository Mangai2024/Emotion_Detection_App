Emotion Detection using CNN
Project Overview

This project is a deep learning–based web application that detects human emotions from facial images.
It uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset and is deployed as an interactive app using Streamlit on Hugging Face Spaces.

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

Data preprocessing of FER-2013 dataset

Building the CNN architecture

Training and evaluating the model

Saving trained model weights

Creating Streamlit web interface

Deploying the app on Hugging Face

Model Architecture

Convolutional layers for feature extraction

MaxPooling layers for downsampling

Fully connected dense layers

Dropout for overfitting prevention

Softmax output layer for emotion classification

Key Features

Upload a face image

Real-time emotion prediction

Confidence score display

Clean and interactive UI

Public deployment on Hugging Face

Live Demo

Try the app here:
https://huggingface.co/spaces/Mangai2024/Emotion_Detection

Challenges Faced

Model file path mismatch during deployment

TensorFlow and Python version conflicts

Keras model deserialization errors

Solved by rebuilding the model architecture and loading weights only

Future Improvements

Real-time webcam emotion detection

Multi-face detection support

Model accuracy enhancement

Mobile-friendly UI

Author

Mangai
Aspiring Data Scientist / AI Engineer
Open to Data Science, ML, and AI opportunities
