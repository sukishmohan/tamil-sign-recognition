# app.py
# Flask web application for real-time Tamil Sign Language Recognition
# Run with:  python app.py
# Then open  http://localhost:5000  in your browser

import os
import sys
import time
import threading
from collections import deque, Counter
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, render_template, request, send_file
from gtts import gTTS

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils import IDX_TO_CHAR, SORTED_FOLDERS, is_background

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(PROJECT_DIR, "models")
MODEL_PATH      = os.path.join(MODELS_DIR, "tamil_sign_model.keras")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
CLASS_MAP_PATH  = os.path.join(MODELS_DIR, "class_names.npy")

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE         = (224, 224)
CONFIDENCE_THRES = 0.65     # minimum model confidence to accept prediction
VOTE_WINDOW      = 20        # frames for majority-vote smoothing
COLLECT_HOLD_SEC = 2.0      # seconds to hold a sign before letter is added
COLLECT_STEPS    = 7        # visual steps shown in the ring (1/7 … 7/7)
CAMERA_INDEX     = 0

app = Flask(__name__)

# ── Disable browser caching for dev mode ──────────────────────────────────────
@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ── Shared state (protected by lock) ─────────────────────────────────────────
_lock = threading.Lock()
_state = {
    "letter":        "",
    "confidence":    0.0,
    "is_background": True,
    "sequence":      [],        # list of Tamil chars added so far
    "collect_step":  0,
    "collect_target": COLLECT_STEPS,
    "fps":           0.0,
    "status":        "Starting…",
}
_frame_jpeg = None   # latest JPEG bytes for MJPEG stream
_running    = False


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    for path in [MODEL_PATH, BEST_MODEL_PATH]:
        if os.path.exists(path):
            print(f"[INFO] Loading model: {path}")
            m = tf.keras.models.load_model(path)
            print("[OK]  Model loaded.")
            return m
    raise FileNotFoundError("No trained model found in ./models/. Run train_model.py first.")


def load_class_map():
    if os.path.exists(CLASS_MAP_PATH):
        names = list(np.load(CLASS_MAP_PATH, allow_pickle=True))
        return {i: name for i, name in enumerate(names)}
    # Fallback: class_indices.npy
    ci_path = os.path.join(MODELS_DIR, "class_indices.npy")
    if os.path.exists(ci_path):
        try:
            ci = np.load(ci_path, allow_pickle=True).item()
            if isinstance(ci, dict):
                return {v: k for k, v in ci.items()}
        except Exception:
            pass
    return {i: folder for i, folder in enumerate(SORTED_FOLDERS)}


# ── Frame preprocessing ───────────────────────────────────────────────────────

def preprocess(frame: np.ndarray) -> np.ndarray:
    h, w  = frame.shape[:2]
    side  = min(h, w)
    y0    = (h - side) // 2
    x0    = (w - side) // 2
    roi   = frame[y0:y0 + side, x0:x0 + side]
    roi   = cv2.resize(roi, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    roi   = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi   = roi.astype(np.float32) / 255.0
    return np.expand_dims(roi, axis=0)


# ── Capture + inference thread ────────────────────────────────────────────────

def capture_loop(camera_idx: int = CAMERA_INDEX):
    global _frame_jpeg, _running

    model         = load_model()
    idx_to_folder = load_class_map()

    # Warm-up
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("[OK]  Model warmed up.")

    # ── OpenCV face detector (to block face false-positives) ─────────────
    _face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    print("[OK]  Face cascade loaded.")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        with _lock:
            _state["status"] = f"Cannot open camera {camera_idx}"
        print(f"[ERROR] Cannot open camera index {camera_idx}.")
        _running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Majority voter
    vote_buffer  = deque(maxlen=VOTE_WINDOW)

    # Time-based collect state
    collect_char  = ""
    collect_start = 0.0   # timestamp when current letter was first seen stably

    fps_counter  = 0
    fps_timer    = time.time()
    fps          = 0.0

    _running = True
    print("[OK]  Camera open. Starting inference loop.")

    while _running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)   # mirror for natural interaction
        now   = time.time()

        # ── Face detection — skip inference if only a face is visible ────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        # A face is considered "dominant" if there is a detected face
        # and the frame has no likely hand region (skin area outside face bbox)
        face_dominant = False
        if len(faces) > 0:
            h_f, w_f = frame.shape[:2]
            # Build a mask of the face regions
            face_mask = np.zeros((h_f, w_f), dtype=np.uint8)
            for (fx, fy, fw, fh) in faces:
                face_mask[fy:fy+fh, fx:fx+fw] = 255

            # Detect skin pixels (YCrCb skin range)
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb,
                                    np.array([0,  133, 77]),
                                    np.array([255, 173, 127]))
            # Skin outside face regions
            hand_skin = cv2.bitwise_and(skin_mask,
                                        cv2.bitwise_not(face_mask))
            hand_skin_px = int(np.count_nonzero(hand_skin))
            # If skin outside face < 2000 px, assume no hand is present
            face_dominant = (hand_skin_px < 2000)

        # ── Model inference (skip when face dominates) ──────────────────
        conf       = 0.0
        tamil_char = ""
        is_bg      = True

        if not face_dominant:
            blob  = preprocess(frame)
            preds = model.predict(blob, verbose=0)[0]
            raw   = int(np.argmax(preds))
            conf  = float(preds[raw])

            vote_buffer.append(raw)
            smoothed    = Counter(vote_buffer).most_common(1)[0][0]
            folder_name = idx_to_folder.get(smoothed, "Background")
            is_bg       = (folder_name == "Background") or (conf < CONFIDENCE_THRES)
            tamil_char  = "" if is_bg else IDX_TO_CHAR.get(smoothed, "?")
        else:
            # Face detected, no hand — flush stale votes
            vote_buffer.clear()

        # ── Time-based collect-to-add ──────────────────────────────────
        collect_step    = 0
        collect_elapsed = 0.0

        if face_dominant or is_bg or tamil_char == "":
            collect_char  = ""
            collect_start = 0.0
        else:
            if tamil_char != collect_char:
                collect_char  = tamil_char
                collect_start = now

            collect_elapsed  = now - collect_start
            collect_fraction = min(collect_elapsed / COLLECT_HOLD_SEC, 1.0)
            collect_step     = int(collect_fraction * COLLECT_STEPS)

            if collect_fraction >= 1.0:
                with _lock:
                    _state["sequence"].append(tamil_char)
                collect_char  = ""
                collect_start = 0.0
                collect_step  = 0

        # ── FPS ────────────────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps         = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.time()

        # ── Status text ────────────────────────────────────────────────
        if face_dominant:
            status = "Face detected — show your hand"
        elif is_bg:
            status = "No sign detected"
        elif collect_step > 0:
            remaining = round(COLLECT_HOLD_SEC - collect_elapsed, 1)
            status = f"Collecting {collect_step}/{COLLECT_STEPS} — hold still ({remaining}s)"
        else:
            status = "Hold the sign steady…"

        # ── Update shared state ────────────────────────────────────────
        with _lock:
            _state["letter"]        = tamil_char
            _state["confidence"]    = conf if not is_bg else 0.0
            _state["is_background"] = is_bg or face_dominant
            _state["collect_step"]  = collect_step
            _state["collect_target"] = COLLECT_STEPS
            _state["fps"]           = round(fps, 1)
            _state["status"]        = status

        # ── Draw ROI box on frame and encode JPEG ──────────────────────
        h, w = frame.shape[:2]
        side  = min(h, w)
        y0    = (h - side) // 2
        x0    = (w - side) // 2
        # Dashed blue rectangle (drawn as segments)
        dash_len = 20
        gap_len  = 14
        color    = (220, 100, 50)   # BGR ≈ blue-purple
        thickness = 3
        pts = []
        # Top & bottom edges
        for edge_y in [y0, y0 + side]:
            x = x0
            drawing = True
            while x < x0 + side:
                end_x = min(x + dash_len, x0 + side)
                if drawing:
                    pts.append(((x, edge_y), (end_x, edge_y)))
                x += dash_len + gap_len
                drawing = not drawing
        # Left & right edges
        for edge_x in [x0, x0 + side]:
            y = y0
            drawing = True
            while y < y0 + side:
                end_y = min(y + dash_len, y0 + side)
                if drawing:
                    pts.append(((edge_x, y), (edge_x, end_y)))
                y += dash_len + gap_len
                drawing = not drawing
        for p1, p2 in pts:
            cv2.line(frame, p1, p2, color, thickness)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _frame_jpeg = buf.tobytes()

    cap.release()
    print("[INFO] Camera released.")


# ── MJPEG generator ───────────────────────────────────────────────────────────

def generate_frames():
    while True:
        with _lock:
            frame = _frame_jpeg
        if frame is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.01)


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/state")
def state():
    with _lock:
        return jsonify(dict(_state))


@app.route("/clear", methods=["POST"])
def clear():
    with _lock:
        _state["sequence"] = []
    return jsonify({"ok": True})


@app.route("/speak/<letter>")
def speak(letter):
    """Generate Tamil speech for a single letter using gTTS."""
    print(f"[SPEAK] Request for letter: {letter}")
    if not letter or letter == "—":
        print(f"[SPEAK] Invalid letter: {letter}")
        return jsonify({"error": "Invalid letter"}), 400
    try:
        print(f"[SPEAK] Generating gTTS audio for: {letter}")
        tts = gTTS(text=letter, lang="ta", slow=False)
        audio_buf = BytesIO()
        tts.write_to_fp(audio_buf)
        audio_buf.seek(0)
        print(f"[SPEAK] Audio generated, size: {len(audio_buf.getvalue())} bytes")
        return send_file(audio_buf, mimetype="audio/mpeg")
    except Exception as e:
        print(f"[ERROR] gTTS failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start capture thread before Flask
    t = threading.Thread(target=capture_loop, args=(CAMERA_INDEX,), daemon=True)
    t.start()

    print("\n" + "=" * 55)
    print("  Tamil Sign Recognition — Web App")
    print("  Open  http://localhost:5000  in your browser")
    print("=" * 55 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
