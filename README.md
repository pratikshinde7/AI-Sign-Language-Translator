AI Sign Language Translator 🤟
<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
</p>

This project is a real-time American Sign Language (ASL) translator that uses a computer's webcam to recognize hand gestures and translate them into text. The application is built with Python, utilizing OpenCV for computer vision and a custom-trained Keras/TensorFlow model for gesture classification.

## Features
Real-Time Translation: Instantly translates ASL gestures captured from a webcam.

Comprehensive Alphabet: Recognizes all 26 letters of the English alphabet.

Special Characters: Includes recognition for the "I Love You" sign.

User-Friendly Interface: Displays the live camera feed with an overlay showing the predicted letter and its confidence score.

## How It Works
The project follows a standard machine learning pipeline:

Data Collection (data_collection.py): A custom script captures images of hand gestures via webcam. It uses the cvzone library to detect hands and isolates them with a bounding box.

Image Preprocessing: Each captured hand gesture is cropped and resized to a uniform 300x300 pixel format while maintaining its original aspect ratio. The hand is centered on a white background to create consistent training data.

Model Training: A Convolutional Neural Network (CNN) was trained on the collected dataset using Keras with a TensorFlow backend. The model learns to classify the 27 different sign language gestures.

Real-Time Detection (run_translator.py): The main application script loads the pre-trained model. It processes the live webcam feed using the same preprocessing steps and feeds the normalized hand gesture image into the model to get a prediction, which is then displayed on the screen.
