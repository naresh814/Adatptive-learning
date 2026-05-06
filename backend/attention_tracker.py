"""
attention_tracker.py — Optimized standalone tracker
====================================================
FIXES: frame-skip, smoothing buffer, confidence gate
"""

import cv2
import numpy as np
import collections
import time
from tensorflow.keras.models import load_model

# Load once
model   = load_model("../model/emotion_model.h5")
EMOTIONS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
ATTENTION_MAP = {"Happy":90,"Neutral":78,"Surprise":72,"Sad":40,"Fear":30,"Angry":25,"Disgust":20}

FRAME_SKIP    = 2
SMOOTH_WINDOW = 7
MIN_CONF      = 0.40

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

emotion_buf    = collections.deque(maxlen=SMOOTH_WINDOW)
attention_score = 100
frame_count    = 0
last_emotion   = "Neutral"
fps_t          = time.time()
fps_f          = 0
fps            = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    fps_f       += 1

    if time.time() - fps_t >= 1.0:
        fps   = fps_f
        fps_f = 0
        fps_t = time.time()

    small    = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    gray_sm  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_sm  = cv2.equalizeHist(gray_sm)

    faces_sm = face_cascade.detectMultiScale(gray_sm, 1.2, 4, minSize=(20,20))
    faces    = [(int(x*2),int(y*2),int(w*2),int(h*2)) for (x,y,w,h) in faces_sm] if len(faces_sm) else []

    status = "Not Detected"

    if faces and (frame_count % FRAME_SKIP == 0):
        x, y, w, h = faces[0]
        gray_full  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi        = gray_full[y:y+h, x:x+w]
        roi        = cv2.resize(roi, (48,48)).astype("float32") / 255.0
        inp        = roi.reshape(1,48,48,1)

        preds      = model.predict(inp, verbose=0)[0]
        idx        = np.argmax(preds)
        conf       = float(preds[idx])

        if conf >= MIN_CONF:
            last_emotion = EMOTIONS[idx]
            emotion_buf.append(last_emotion)
        status = "Focused"
    elif not faces:
        attention_score = max(0, attention_score - 1)
        status = "Distracted"

    smoothed = collections.Counter(emotion_buf).most_common(1)[0][0] if emotion_buf else "Neutral"
    attn_val = ATTENTION_MAP.get(smoothed, 50)

    cv2.putText(frame, f"Attention: {attention_score}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Status: {status}",             (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"Emotion: {smoothed}",          (20,120),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,165,0),2)
    cv2.putText(frame, f"FPS: {fps}",                   (20,160),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,0),2)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Attention Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
