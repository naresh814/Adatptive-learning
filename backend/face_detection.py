"""
face_detection.py — Optimized standalone face detection test
"""
import cv2, time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

fps_t, fps_f, fps = time.time(), 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    fps_f += 1
    if time.time() - fps_t >= 1.0:
        fps = fps_f; fps_f = 0; fps_t = time.time()

    # Downscale for faster detection
    small    = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    gray_sm  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_sm  = cv2.equalizeHist(gray_sm)
    faces_sm = face_cascade.detectMultiScale(gray_sm, 1.2, 4, minSize=(20,20))

    for (x,y,w,h) in faces_sm:
        x2,y2,w2,h2 = int(x*2),int(y*2),int(w*2),int(h*2)
        cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)

    label = "Face Detected" if len(faces_sm) > 0 else "No Face"
    color = (0,255,0) if len(faces_sm) > 0 else (0,0,255)
    cv2.putText(frame, label,     (20,40),  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"FPS:{fps}",(20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,0),2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
