import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import random

#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    # if img != []:
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     face_points = face_detector.detectMultiScale(gray, 1.3, 5)

    #     # Apply image augmentation to capture multiple images of the face
    #     augmented_faces = []
    #     for face_point in face_points:
    #         try:
    #             x, y, w, h = face_point
    #             face = img[y:y+h, x:x+w]
    #             augmented_faces.append(face)

    #             # Flip the image horizontally
    #             flipped_face = cv2.flip(face, 1)
    #             augmented_faces.append(flipped_face)

    #             # Rotate the image
    #             (h, w) = face.shape[:2]
    #             center = (w // 2, h // 2)
    #             angle = 30
    #             M = cv2.getRotationMatrix2D(center, angle, 1.0)
    #             rotated_face = cv2.warpAffine(face, M, (w, h))
    #             augmented_faces.append(rotated_face)
    #         except:
    #             pass

    #     return augmented_faces
    # else:
    #     return []


    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            
            # perform image augmentation
            flip = random.randint(0, 1)
            if flip:
                img = cv2.flip(img, 1)
            angle = random.uniform(-15, 15)
            scale = random.uniform(0.8, 1.2)
            rotation_matrix = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, scale)
            img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
            
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')