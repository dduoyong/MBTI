import cv2
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

cascade_path = './emotion/haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)
emotion_dict= {0:'moved', 1:'fearful', 2:'surprised', 3:'angry', 4:'anxious', 5:'smitten', 6:'fluttered',
               7:'disappointed', 8:'fulfilled', 9:'ashamed', 10:'sad', 11:'upset', 12:'sympathetic', 13:'passionate',
               14:'depressed', 15:'amused', 16:'affectionate', 17:'guilty', 18:'jealous', 19:'peaceful', 20:'happy', 21:'disgusted'}

emt = [0]
"""Load model from json and h5 file"""
model = model_from_json(open("./emotion/models/22emo_1st_model_accuracy_81.json", "r").read())
model.load_weights('./emotion/models/22emo_1st_model_accuracy_81.h5')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)

flag = True
while flag:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (650, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
      roi_gray = gray[y:y + h, x:x + w]
      input = np.expand_dims(np.expand_dims(cv2.resize(frame, (64, 64)), -1), 0)
      prediction = model.predict(input)
      idx = np.argmax(prediction)
      emt[0] = idx

      cv2.rectangle(frame, (x, y), (x + w, y + h), (124, 252, 0), 2)
      cv2.putText(frame, emotion_dict[idx], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    last_frame = frame.copy()

    cv2.imshow('VideoFrame', frame)
    time.sleep(1 / 30)
    key = cv2.waitKey(33)
    if key == 27:
        flag = False
cap.release()
