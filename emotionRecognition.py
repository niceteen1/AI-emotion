import numpy as np
import argparse
import dlib
import cv2
from keras.models import *
import pygame
import random
import mediapipe as mp

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

emotion_offsets = (20, 40)
emotions = {
    0: {
        "emotion": "Angry",
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",
        "color": (108, 72, 200)
    }
}

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

faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

emotionModelPath = 'models/emotionModel.hdf5'  
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

cap = cv2.VideoCapture(0)

if args["isVideoWriter"]:
    fourrcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    capWidth = int(cap.get(3))
    capHeight = int(cap.get(4))
    videoWrite = cv2.VideoWriter("output.avi", fourrcc, 22, (capWidth, capHeight))

pygame.mixer.init()
music_files = {
    "Angry": ['music/Angry/1.mp3', 'music/Angry/2.mp3', 'music/Angry/3.mp3', 'music/Angry/4.mp3', 'music/Angry/5.mp3'],
    "Happy": ['music/Happy/1.mp3', 'music/Happy/2.mp3', 'music/Happy/3.mp3', 'music/Happy/4.mp3', 'music/Happy/5.mp3'],
    "Suprise": ['music/Suprise/SUR1.mp3','music/Suprise/SUR2.mp3','music/Suprise/SUR3.mp3','music/Suprise/SUR4.mp3','music/Suprise/SUR5.mp3'],
    "Sad": ['music/Sad/1.mp3','music/Sad/2.mp3','music/Sad/3.mp3','music/Sad/4.mp3','music/Sad/SAD1.mp3','music/Sad/SAD2.mp3','music/Sad/SAD4.mp3','music/Sad/SAD5.mp3','music/Sad/SAD3.mp3']

}
current_emotion = None
is_music_playing = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
hand_count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for rect in rects:
        shape = predictor(grayFrame, rect)
        points = shapePoints(shape)
        (x, y, w, h) = rectPoints(rect)
        grayFace = grayFrame[y:y + h, x:x + w]
        try:
            grayFace = cv2.resize(grayFace, (emotionTargetSize))
        except:
            continue

        grayFace = grayFace.astype('float32')
        grayFace = grayFace / 255.0
        grayFace = (grayFace - 0.5) * 2.0
        grayFace = np.expand_dims(grayFace, 0)
        grayFace = np.expand_dims(grayFace, -1)
        emotion_prediction = emotionClassifier.predict(grayFace)
        emotion_probability = np.max(emotion_prediction)
        if (emotion_probability > 0.36):
            emotion_label_arg = np.argmax(emotion_prediction)
            color = emotions[emotion_label_arg]['color']
            emotion = emotions[emotion_label_arg]['emotion']
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.line(frame, (x, y + h), (x + 20, y + h + 20), color, thickness=2)
            cv2.rectangle(frame, (x + 20, y + h + 20), (x + 110, y + h + 40), color, -1)
            cv2.rectangle(frame, (x + 20, y + h + 20), (x + 110, y + h + 40), color, -1)
            cv2.putText(frame, emotion, (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if emotion in music_files:
                if emotion != current_emotion:
                    pygame.mixer.music.stop()
                    music_file = random.choice(music_files[emotion])
                    pygame.mixer.music.load(music_file)
                    pygame.mixer.music.play(-1)
                    current_emotion = emotion
                    is_music_playing = True
            else:
                if is_music_playing:
                    pygame.mixer.music.stop()
                    current_emotion = None
                    is_music_playing = False    
        else:
            color = (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if is_music_playing:
                pygame.mixer.music.stop()
                is_music_playing = False

    # Process hands
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgbFrame)
    if hand_results.multi_hand_landmarks:
        hand_count = len(hand_results.multi_hand_landmarks)
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if hand_count == 2:  # Check two hands 
        break

    if args["isVideoWriter"]:
        videoWrite.write(frame)

    cv2.imshow("Emotion and Hand Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
if args["isVideoWriter"]:
    videoWrite.release()
cv2.destroyAllWindows()
pygame.mixer.quit()