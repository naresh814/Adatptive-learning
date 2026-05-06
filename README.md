# 🧠 ASD Adaptive Learning System

A real-time personalized learning platform for children with **Autism Spectrum Disorder (ASD)**.  
Detects facial emotions via webcam using a trained CNN model and automatically adapts learning content — difficulty, tone, and pace — in real time.

---

## ✨ Features

- 😊 Real-time facial emotion detection (FER2013 CNN — 7 emotions)
- 🎯 Attention score calculated from detected emotion
- ⚡ Rule-based contextual content selection (Visual / Calm / Break modes)
- 📊 Live monitoring dashboard with Chart.js
- 🔊 Voice tone analysis — Wav2Vec2 (Phase 2)
- 📈 Deep Knowledge Tracing — LSTM (Phase 2)
- 📁 Session logging and CSV export

---

## 🚀 Performance Optimizations

| Issue | Fix Applied |
|---|---|
| FPS stuck at ~10 | Frame-skip (every 2nd frame), downscale for detection, buffer=1 |
| Flickering predictions | 7-frame majority-vote smoothing buffer |
| Low confidence | Softmax temperature scaling + min 40% confidence gate |
| Model loading slow | Loaded once at startup + warm-up call |
| High CPU usage | `time.sleep(0.01)` yield + threaded capture |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Emotion Model | CNN — FER2013 (TensorFlow/Keras) |
| Face Detection | OpenCV Haar Cascade |
| Backend | Flask (Python) REST API |
| Frontend | HTML · CSS · Chart.js |
| Smoothing | Majority vote (collections.deque) |
| Threading | Python threading + Lock |

---

## 📁 Project Structure

```
FYP-ADAPTIVE-LEARNING/
├── backend/
│   ├── app.py                  ← Flask backend (OPTIMIZED)
│   ├── train_model.py          ← CNN training (with augmentation + callbacks)
│   ├── attention_tracker.py    ← Standalone tracker
│   ├── content_selector.py     ← Rule-based content policy
│   ├── face_detection.py       ← Face detection test
│   ├── dataset_loader.py       ← Dataset class lister
│   ├── dataset_stats.py        ← Class distribution analyser
│   ├── camera_test.py          ← Camera connectivity test
│   └── webcam.py               ← Simple webcam test
├── data/
│   └── engagement_data.csv     ← Real session records (111)
├── model/                      ← Place emotion_model.h5 here (from Drive)
├── dataset/                    ← Place FER-2013 here (from Drive)
├── static/
│   └── style.css
├── templates/
│   └── index.html              ← Live dashboard
├── .gitignore
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YourName/FYP-Adaptive-Learning.git
cd FYP-Adaptive-Learning
```

### 2. Download large files (model + dataset)
🔗 **[Google Drive — emotion_model.h5 + FER2013 Dataset](PASTE_DRIVE_LINK_HERE)**

Place them:
- `model/emotion_model.h5`
- `dataset/fer2013/train/<emotion>/`

### 3. Activate virtual environment
```bash
cd backend
tf_env\Scripts\activate        # Windows
source tf_env/bin/activate     # Mac / Linux
```

### 4. Install dependencies
```bash
pip install flask tensorflow opencv-python numpy scikit-learn
```

### 5. Run
```bash
python app.py
```

Open **http://127.0.0.1:5000** → Click **▶ Start Session**

---

## 👥 Team

| Name | Role |
|---|---|
| Naresh Kumar | Problem Statement & Objectives |
| Tanish Dogra | Literature Survey |
| Arun Kumar | System Design Methodology |
| Souvik | Implementation & Live Demo |

---

## 🏫 Institution

**NITTE Meenakshi Institute of Technology**  
Department of Information Science & Engineering  
Project Phase II · March 2026
