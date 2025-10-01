# run_translator.py

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model
import os

# Initialize Webcam and Hand Detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Image processing parameters to match your new model
offset = 20
imgSize = 64

# Load your newly trained model
model = load_model('my_model.h5')

# Dynamically load labels from your data folders
labels = sorted(os.listdir('MyData'))

while True:
    success, img = cap.read()
    if not success:
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a square white background (grayscale)
        imgWhite = np.ones((imgSize, imgSize), np.uint8) * 255
        
        # 1. Crop the hand from the original color image
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # 2. Check if the cropped image is valid before proceeding
        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            continue
        
        # 3. NOW convert the valid cropped image to grayscale
        imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

        # Resize the cropped image while maintaining aspect ratio
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Prepare image for prediction in the correct shape
        img_for_prediction = (imgWhite / 255.0).reshape(1, 64, 64, 1)
        
        # Make a prediction
        prediction = model.predict(img_for_prediction, verbose=0) # Added verbose=0 to clean up terminal
        index = np.argmax(prediction)
        predicted_label = labels[index]
        confidence = prediction[0][index]

        # Display the prediction on the screen
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                      (x - offset + 290, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, f"{predicted_label} ({confidence*100:.1f}%)",
                    (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

    cv2.imshow("Sign Language Translator", imgOutput)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
