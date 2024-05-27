# Importing required packages
from mtcnn.mtcnn import MTCNN
import numpy as np
import dlib
import cv2


def putText(frame, text, x, y):
    color = (193, 69, 42) 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    thickness = 2
    fontScale = 0.75 
    org = (x, y) 
    lineType = cv2.LINE_AA 
    cv2.putText(frame, text, org, font, fontScale , color, thickness, lineType)
    return frame


def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

modelFile = "models/dnn/res10_300x300_ssd_iter_140000.caffemodel" 
configFile = "models/dnn/deploy.prototxt" 
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detectFacesWithDNN(frame):
  
    size = (300,300)

    scalefactor = 1.0 

    swapRB = (104.0, 117.0, 123.0)

    height, width = frame.shape[:2]
    resizedFrame = cv2.resize(frame, size)
    blob = cv2.dnn.blobFromImage(resizedFrame, scalefactor, size, swapRB)
    net.setInput(blob)
    dnnFaces = net.forward()
    for i in range(dnnFaces.shape[2]):
        confidence = dnnFaces[0, 0, i, 2]
        if confidence > 0.5:
            box = dnnFaces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x1, y1), (193, 69, 42), 2)
            frame = putText(frame, "DNN", x+5, y-5)
    return frame


faceLandmarks = "models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

def detectFacesWithDLIB(frame):    
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for rect in rects:
        shape = predictor(grayFrame, rect)
        points = shapePoints(shape)
        (x, y, w, h) = rectPoints(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (193, 69, 42), 2)
        frame = putText(frame, "DLIB", x+5, y-5)
    return frame



faceCascade = cv2.CascadeClassifier("models/haarcascade/haarcascade_frontalface_default.xml")

def detectFacesWithCascade(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayFrame, 1.3, 5)
    for faceCoordinates in faces:
        x, y, w, h = faceCoordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (193, 69, 42), 2)
        frame = putText(frame, "Haarcascade", x+5, y-5)
    return frame



mtcnnDetector = MTCNN()

def detectFacesWithMTCNN(frame):
    faces = mtcnnDetector.detect_faces(frame)
    for face in faces:
        if face["confidence"] > 0.5:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (193, 69, 42), 2)
            frame = putText(frame, "MTCNN", x+5, y-5)
    return frame