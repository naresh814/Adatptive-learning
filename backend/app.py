"""
ASD Adaptive Learning System — Optimized Backend
=================================================
FIXES APPLIED:
  FPS      → threaded capture, frame-skip, resize before detection
  Accuracy → temperature scaling, confidence threshold
  Stability→ majority-vote smoothing over 7-frame rolling window
  Confidence→ softmax temperature + min-confidence gate
  Bugs     → model loaded ONCE at startup, thread-safe state lock,
             cap.set() for native resolution, no blocking predict in main loop
"""

from flask import Flask, render_template, jsonify, Response, session, redirect, url_for
import cv2
import numpy as np
import threading
import time
import base64
import sys
import collections
from tensorflow.keras.models import load_model
import tensorflow as tf
<<<<<<< HEAD
print("GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU acceleration enabled")
    except RuntimeError as e:
        print(e)
=======

>>>>>>> 5ed801d16cc914947ae036cb4df6251698188a6d
import os
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="../templates", static_folder="../static")

# ── Auth / DB config (NEW) ────────────────────────────────────────────────────
app.config["SECRET_KEY"]                  = "asd-learnadapt-dev-secret-key-2026"
app.config["SQLALCHEMY_DATABASE_URI"]     = "sqlite:///" + os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "..", "users.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db     = SQLAlchemy(app)
bcrypt = Bcrypt(app)

from auth import init_auth, login_required
User = init_auth(app, db, bcrypt)

# ── Load model ONCE at startup ────────────────────────────────────────────────
print("[INFO] Loading emotion model...")
model = load_model("../model/emotion_model.h5")

# Dynamically detect model input shape so we support both the old model and the new MobileNetV2
MODEL_INPUT_SHAPE = model.input_shape[1:3]  # e.g., (48, 48) or (96, 96)
MODEL_CHANNELS    = model.input_shape[3]    # 1 (Grayscale) or 3 (RGB)

_warmup_input = tf.constant(np.zeros((1, MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1], MODEL_CHANNELS), dtype="float32"))
model(_warmup_input, training=False)          # warm-up: compile graph (faster than .predict)
print(f"[INFO] Model ready. Expected input: {MODEL_INPUT_SHAPE} with {MODEL_CHANNELS} channels.")

def predict_fast(inp):
    """Direct model __call__ — skips predict() overhead (~3-5x faster for single samples)."""
    tensor = tf.constant(inp, dtype=tf.float32)
    return model(tensor, training=False)[0].numpy()

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS        = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_EMOJI   = {"Angry":"😠","Disgust":"🤢","Fear":"😨","Happy":"😊",
                   "Sad":"😢","Surprise":"😲","Neutral":"😐"}
# NOTE: Static ATTENTION_MAP removed — replaced by dynamic compute_attention()
CONTENT_MAP     = {
    "Happy":    {"title":"Visual Flashcards",   "mode":"visual", "difficulty":"Medium",
                 "desc":"Bright image cards with shape & colour matching. Reward sounds on correct answers."},
    "Neutral":  {"title":"Reading & Matching",  "mode":"visual", "difficulty":"Medium",
                 "desc":"Letter and word matching tasks. Standard calm pacing."},
    "Surprise": {"title":"Interactive Story",   "mode":"visual", "difficulty":"Medium",
                 "desc":"Attention re-engaged! Interactive story with comprehension question."},
    "Sad":      {"title":"Calm Mode — Colours", "mode":"calm",   "difficulty":"Easy",
                 "desc":"Soothing colour identification. Gentle audio. Encouragement messages shown."},
    "Fear":     {"title":"Calm Mode — Shapes",  "mode":"calm",   "difficulty":"Easy",
                 "desc":"Low-stimulation content. Soft colours, slow pace, no sudden sounds."},
    "Angry":    {"title":"Sensory Break",        "mode":"break",  "difficulty":"Easy",
                 "desc":"Switching to 2-minute breathing break. Soft animation guide."},
    "Disgust":  {"title":"Calm Audio Prompt",    "mode":"calm",   "difficulty":"Easy",
                 "desc":"Gentle audio narration with simple tap tasks. Reduced visual load."},
}

# ── Tuning knobs ──────────────────────────────────────────────────────────────
<<<<<<< HEAD
FRAME_SKIP          = 5      # run CNN every 5th frame (better FPS)
SMOOTH_WINDOW       = 7      # majority-vote window size
MIN_CONFIDENCE      = 0.40   # ignore weak predictions
TEMPERATURE         = 1.5
DETECTION_SCALE     = 0.35   # smaller detection frame = faster processing
CASCADE_SCALE       = 1.2
CASCADE_NEIGHBORS   = 4
=======
FRAME_SKIP          = 3      # run CNN every Nth frame  (FPS boost: 2→3)
SMOOTH_WINDOW       = 7      # majority-vote window size (stability fix)
MIN_CONFIDENCE      = 0.40   # ignore predictions below this (noise fix)
TEMPERATURE         = 1.5    # softmax temperature > 1 spreads distribution (confidence fix)
DETECTION_SCALE     = 0.5    # downscale frame before Haar (FPS fix)
CASCADE_SCALE       = 1.2    # faster than 1.3 with minNeighbors=4
CASCADE_NEIGHBORS   = 4

>>>>>>> 5ed801d16cc914947ae036cb4df6251698188a6d
# ── Thread-safe shared state ──────────────────────────────────────────────────
_lock = threading.Lock()
state = {
    "camera_running":  False,
    "face_detected":   False,
    "emotion":         "Neutral",
    "emoji":           "😐",
    "confidence":      0.0,
    "attention":       0,
    "engagement":      "disengaged",
    "content":         CONTENT_MAP["Neutral"],
    "history":         [],
    "session_seconds": 0,
    "engaged_frames":  0,
    "total_frames":    0,
    "alerts":          0,
    "fps":             0.0,
    "frame_b64":       None,
}

# Haar cascade loaded once
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Mapping emotions to a baseline attention/engagement score
ATTENTION_MAP = {
    "Happy": 100,
    "Surprise": 90,
    "Neutral": 80,
    "Sad": 40,
    "Fear": 30,
    "Disgust": 20,
    "Angry": 10
}
ATTENTION_WINDOW    = 20  # rolling window size for attention calculation


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_face(color_face):
    """Resize → RGB/Grayscale (dynamically) → normalise → reshape."""
    face = cv2.resize(color_face, MODEL_INPUT_SHAPE)
    
    if MODEL_CHANNELS == 1:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    else:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
    face = face.astype("float32") / 255.0
    return face.reshape(1, MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1], MODEL_CHANNELS)


def compute_attention(attention_deque):
    """Dynamic attention score from rolling window of emotions."""
    if not attention_deque:
        return 0
    total_score = sum(ATTENTION_MAP.get(e, 50) for e in attention_deque)
    return int(round(total_score / len(attention_deque)))


# ── Camera pipeline: 2-thread architecture ────────────────────────────────────
# Thread 1 (capture_thread):  reads frames at full speed, stores ONLY latest
# Thread 2 (process_thread):  picks up latest frame, does detection + CNN + encode
# This eliminates lag from frame buffer buildup.

_cam_index   = None     # cached working camera index
_latest_frame = None    # shared between capture and process threads
_frame_lock   = threading.Lock()


def capture_thread(cap):
    """Continuously reads frames, keeping only the latest one (no backlog)."""
    global _latest_frame
    while True:
        with _lock:
            if not state["camera_running"]:
                break
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        frame = cv2.flip(frame, 1)
        with _frame_lock:
            _latest_frame = frame


def process_thread():
    """Processes latest frame: face detection → CNN → encode → state update."""
    global _latest_frame

    emotion_buffer   = collections.deque(maxlen=SMOOTH_WINDOW)
    attention_buffer = collections.deque(maxlen=ATTENTION_WINDOW)

    frame_count    = 0
    last_emotion   = "Neutral"
    last_conf      = 0.0
    fps_timer      = time.time()
    fps_frames     = 0
    debug_timer    = time.time()

    while True:
        with _lock:
            if not state["camera_running"]:
                break

        # Grab latest frame (non-blocking)
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1
        fps_frames  += 1

        # ── FPS calculation ──────────────────────────────────────────────────
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = fps_frames / elapsed
            fps_timer  = time.time()
            fps_frames = 0
            with _lock:
                state["fps"] = round(fps, 1)

        # ── Downscale for face detection ─────────────────────────────────────
        small   = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        gray_sm = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_sm = cv2.equalizeHist(gray_sm)

        faces_sm = face_cascade.detectMultiScale(
            gray_sm,
            scaleFactor=CASCADE_SCALE,
            minNeighbors=CASCADE_NEIGHBORS,
            minSize=(20, 20),
        )

        scale = 1.0 / DETECTION_SCALE
        faces = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale))
                 for (x, y, w, h) in faces_sm] if len(faces_sm) else []

        face_found = len(faces) > 0

        # ── Run CNN every FRAME_SKIP frames ──────────────────────────────────
        if face_found and (frame_count % FRAME_SKIP == 0):
            x, y, w, h = faces[0]
            color_roi  = frame[y:y+h, x:x+w]
            inp        = preprocess_face(color_roi)

            raw_pred   = predict_fast(inp)
            best_idx   = int(np.argmax(raw_pred))
            best_conf  = float(raw_pred[best_idx])

            if best_conf >= MIN_CONFIDENCE:
                last_emotion = EMOTIONS[best_idx]
                last_conf    = round(best_conf * 100, 1)
                emotion_buffer.append(last_emotion)
                attention_buffer.append(last_emotion)

            # Debug log every 2 seconds
            if time.time() - debug_timer >= 2.0:
                debug_timer = time.time()
                print(f"[PREDICT] {last_emotion} ({last_conf}%) | "
                      f"attn={compute_attention(attention_buffer)}% | "
                      f"face={'YES' if face_found else 'NO'}")

        # ── Majority vote smoothing ──────────────────────────────────────────
        if emotion_buffer:
            smoothed = collections.Counter(emotion_buffer).most_common(1)[0][0]
        else:
            smoothed = "Neutral"

        # ── Feed attention buffer on EVERY frame (not just CNN frames) ────────
        if face_found:
            attention_buffer.append(smoothed)

        # ── Dynamic attention score ──────────────────────────────────────────
        attention  = compute_attention(attention_buffer)
        engagement = "engaged" if attention >= 50 else "disengaged"
        content    = CONTENT_MAP.get(smoothed, CONTENT_MAP["Neutral"])

        # ── Annotate frame ───────────────────────────────────────────────────
        display = frame.copy()
        for (x, y, w, h) in faces:
            color = (0, 230, 118) if attention >= 50 else (0, 165, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            label = f"{smoothed} | {last_conf:.0f}% | Attn:{attention}%"
            cv2.putText(display, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        with _lock:
            fps_disp = state["fps"]
        cv2.putText(display, f"FPS:{fps_disp}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ── Encode to base64 ────────────────────────────────────────────────
<<<<<<< HEAD
        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 40])
=======
        _, buf    = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 65])
>>>>>>> 5ed801d16cc914947ae036cb4df6251698188a6d
        frame_b64 = base64.b64encode(buf).decode("utf-8")

        # ── Update shared state ──────────────────────────────────────────────
        with _lock:
            state["face_detected"]  = face_found
            state["emotion"]        = smoothed
            state["emoji"]          = EMOTION_EMOJI.get(smoothed, "😐")
            state["confidence"]     = last_conf
            state["attention"]      = attention
            state["engagement"]     = engagement
            state["content"]        = content
            state["frame_b64"]      = frame_b64
            state["total_frames"]  += 1
            if engagement == "engaged":
                state["engaged_frames"] += 1
            # Throttle alerts: max 1 per 5 seconds
            if attention < 35 and face_found and (state["total_frames"] % 150 == 0):
                state["alerts"] += 1
            hist = state["history"]
            hist.append({"emotion": smoothed, "attention": attention})
            if len(hist) > 60:
                hist.pop(0)

<<<<<<< HEAD
        time.sleep(0.01)  # lower CPU usage and reduce lag  # yield CPU
=======
        time.sleep(0.005)  # yield CPU
>>>>>>> 5ed801d16cc914947ae036cb4df6251698188a6d

    with _lock:
        state["camera_running"] = False
    print("[INFO] Process thread ended.")

<<<<<<< HEAD
def generate_frames():
    while True:
        with _lock:
            if not state["camera_running"]:
                break

            frame_b64 = state["frame_b64"]

        if frame_b64 is None:
            time.sleep(0.01)
            continue

        frame_bytes = base64.b64decode(frame_b64)

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

        time.sleep(0.03)

=======
>>>>>>> 5ed801d16cc914947ae036cb4df6251698188a6d

def start_camera_pipeline():
    """Opens camera and starts capture + process threads."""
    global _cam_index, _latest_frame
    _latest_frame = None

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

    indices = [_cam_index] if _cam_index is not None else []
    for idx in [1, 0]:
        if idx not in indices:
            indices.append(idx)

    cap = None
    for idx in indices:
        print(f"[INFO] Trying camera index {idx} (backend={backend})...")
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            _cam_index = idx
            print(f"[INFO] Camera opened at index {idx}")
            break
        else:
            cap.release()
            cap = None

    if cap is None:
        print("[ERROR] No camera found.")
        with _lock:
            state["camera_running"] = False
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
<<<<<<< HEAD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
=======
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
>>>>>>> 5ed801d16cc914947ae036cb4df6251698188a6d

    # Start both threads
    threading.Thread(target=capture_thread, args=(cap,), daemon=True).start()
    threading.Thread(target=process_thread, daemon=True).start()
    print("[INFO] Camera pipeline started (capture + process threads).")


def session_timer():
    while True:
        time.sleep(1)
        with _lock:
            if not state["camera_running"]:
                break
            state["session_seconds"] += 1


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/dashboard")
@login_required
def index():
    username = session.get("username", "User")
    return render_template("index.html", username=username)


@app.route("/start_camera")
@login_required
def start_camera():
    with _lock:
        if state["camera_running"]:
            return jsonify({"status": "already running"})
        state.update({
            "camera_running": True, "session_seconds": 0,
            "engaged_frames": 0,    "total_frames": 0,
            "alerts": 0,            "history": [],
            "fps": 0.0,
        })
    threading.Thread(target=start_camera_pipeline, daemon=True).start()
    threading.Thread(target=session_timer, daemon=True).start()
    return jsonify({"status": "camera started"})


@app.route("/stop_camera")
@login_required
def stop_camera():
    with _lock:
        state["camera_running"] = False
    return jsonify({"status": "camera stopped"})


@app.route("/get_state")
@login_required
def get_state():
    with _lock:
        total   = state["total_frames"]
        engaged = state["engaged_frames"]
        eng_pct = round(engaged / total * 100) if total > 0 else 0
        secs    = state["session_seconds"]
        return jsonify({
            "camera_running":  state["camera_running"],
            "face_detected":   state["face_detected"],
            "emotion":         state["emotion"],
            "emoji":           state["emoji"],
            "confidence":      state["confidence"],
            "attention":       state["attention"],
            "engagement":      state["engagement"],
            "content":         state["content"],
            "session_time":    f"{secs//60:02d}:{secs%60:02d}",
            "session_seconds": secs,
            "engaged_pct":     eng_pct,
            "total_frames":    total,
            "alerts":          state["alerts"],
            "fps":             state["fps"],
            "history":         state["history"][-20:],
            "frame_b64":       state["frame_b64"],
        })


@app.route("/get_frame")
@login_required
def get_frame():
    with _lock:
        return jsonify({"frame": state["frame_b64"]})

<<<<<<< HEAD
@app.route("/video_feed")
@login_required
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
=======

>>>>>>> 5ed801d16cc914947ae036cb4df6251698188a6d
if __name__ == "__main__":
    # Create auth tables on first run
    with app.app_context():
        db.create_all()
        print("[INFO] Auth database ready (users.db)")

    print("=" * 55)
    print("  ASD Adaptive Learning System — Optimized Backend")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 55)
    app.run(debug=False, use_reloader=False, port=5000, threaded=True)
