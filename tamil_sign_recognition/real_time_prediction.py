# real_time_prediction.py
# Real-time Tamil hand sign recognition using a trained model and webcam.
#
# Features:
#   • Smooth predictions via majority voting over a sliding window
#   • Confidence threshold to suppress uncertain predictions
#   • Word formation: detected letters accumulate into a word
#   • Optional: word matching against the README.txt word list
#
# Usage:
#   python real_time_prediction.py
#   python real_time_prediction.py --camera 1        # use camera index 1
#   python real_time_prediction.py --threshold 0.70  # confidence threshold
#   python real_time_prediction.py --window 15       # majority-vote window
#   python real_time_prediction.py --no-display      # terminal only (no GUI)

import os
import sys
import argparse
import time
from collections import deque, Counter

# Force unbuffered stdout so every print() appears immediately in terminal
sys.stdout.reconfigure(line_buffering=True)

# ── Live log file (tail this in a second terminal to see output) ──────────────
_LOG_PATH = os.path.join(os.path.dirname(__file__), "live_output.log")
_log_file = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)

def log(msg: str):
    """Print to terminal AND write to live_output.log simultaneously."""
    print(msg, flush=True)
    _log_file.write(msg + "\n")
    _log_file.flush()

import numpy as np
import cv2
import tensorflow as tf
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    NUM_CLASSES, IDX_TO_CHAR, SORTED_FOLDERS,
    is_background, setup_utf8_console,
    load_word_list,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR     = os.path.dirname(__file__)
MODELS_DIR      = os.path.join(PROJECT_DIR, "models")
MODEL_PATH      = os.path.join(MODELS_DIR, "tamil_sign_model.keras")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
CLASS_MAP_PATH  = os.path.join(MODELS_DIR, "class_indices.npy")
README_PATH     = os.path.join(
    PROJECT_DIR, "..",
    "TLFS23 - Tamil Language Finger Spelling Image Dataset",
    "ReadMe.txt",
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE         = (224, 224)
CONFIDENCE_THRES = 0.65      # Minimum confidence to accept a prediction
VOTE_WINDOW      = 15        # Frames for majority-vote smoothing
ADD_LETTER_DELAY = 1.5       # Seconds a letter must be stable before adding
WORD_GAP_DELAY   = 2.5       # Seconds of background before word is finalized

# ── Overlay colours (BGR) ─────────────────────────────────────────────────────
COL_GREEN  = (0, 220, 0)
COL_RED    = (0, 0, 220)
COL_YELLOW = (0, 200, 200)
COL_WHITE  = (255, 255, 255)
COL_DARK   = (30, 30, 30)
COL_BLUE   = (220, 100, 0)


# ── Tamil font loading ────────────────────────────────────────────────────────
def _load_tamil_font(size: int):
    """Find and load a font that can render Tamil Unicode on Windows."""
    candidates = [
        r"C:\Windows\Fonts\latha.ttf",
        r"C:\Windows\Fonts\NirmalaS.ttf",
        r"C:\Windows\Fonts\nirmala.ttf",
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\leelawad.ttf",
        r"C:\Windows\Fonts\LeelawUI.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return None


if _PIL_OK:
    _FONT_BIG   = _load_tamil_font(34)
    _FONT_SMALL = _load_tamil_font(24)
    if _FONT_BIG is None:
        print("[WARN] No Tamil font found — text may not display correctly.")
else:
    _FONT_BIG = _FONT_SMALL = None
    print("[WARN] Pillow not installed — Tamil text cannot be displayed on screen.")
    print("       Run: pip install pillow")


# ── Load model ────────────────────────────────────────────────────────────────

def load_model() -> tf.keras.Model:
    """Load the trained model; prefers the final model, falls back to best."""
    for path in [MODEL_PATH, BEST_MODEL_PATH]:
        if os.path.exists(path):
            print(f"[INFO] Loading model: {path}")
            model = tf.keras.models.load_model(path)
            print("[OK] Model loaded.")
            return model
    print("[ERROR] No trained model found in ./models/")
    print("        Run  python train_model.py  first.")
    sys.exit(1)


def load_class_map() -> dict:
    """
    Load the class-index → folder-name mapping saved during training.
    Returns {class_index: folder_name_str}.
    Folder names are numeric strings ("1"–"247") or "Background".
    """
    class_names_path = os.path.join(MODELS_DIR, "class_names.npy")
    # Prefer class_names.npy (array of folder name strings saved by get_datasets)
    if os.path.exists(class_names_path):
        names = list(np.load(class_names_path, allow_pickle=True))
        return {i: name for i, name in enumerate(names)}
    # Legacy: class_indices.npy saved as {folder_name: index} dict
    if os.path.exists(CLASS_MAP_PATH):
        try:
            class_indices = np.load(CLASS_MAP_PATH, allow_pickle=True).item()
            if isinstance(class_indices, dict):
                return {v: k for k, v in class_indices.items()}
        except Exception:
            pass
    # Fall back: use the ordering computed in utils.py (string sort)
    return {i: folder for i, folder in enumerate(SORTED_FOLDERS)}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Crop a square ROI from the centre of the frame, resize, and normalise.
    Returns a batch of shape (1, H, W, 3).
    """
    h, w = frame.shape[:2]
    side = min(h, w)
    y0   = (h - side) // 2
    x0   = (w - side) // 2
    roi  = frame[y0:y0 + side, x0:x0 + side]
    roi  = cv2.resize(roi, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    roi  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi  = roi.astype(np.float32) / 255.0
    return np.expand_dims(roi, axis=0)


# ── Overlay helpers ───────────────────────────────────────────────────────────

def draw_roi_box(frame: np.ndarray):
    """Draw the square ROI rectangle on the frame."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), COL_GREEN, 2)


def put_text_safe(frame, text, pos, font_scale=0.8, color=COL_WHITE,
                  thickness=2, bg=True):
    """Draw ASCII text with an optional dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(frame,
                      (x - 4, y - th - 6),
                      (x + tw + 4, y + baseline),
                      COL_DARK, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)


def apply_tamil_overlays(frame: np.ndarray,
                         char_display: str,
                         current_word: str,
                         is_bg: bool,
                         finalized: str = None) -> np.ndarray:
    """
    Draw Tamil Unicode text (letter + word) on frame using PIL.
    All PIL round-trips are batched into one conversion per frame.
    """
    if not _PIL_OK or _FONT_BIG is None:
        # Fallback: show question marks for Tamil chars via cv2
        safe = lambda s: s.encode('ascii', 'replace').decode()
        put_text_safe(frame, f"Letter: {safe(char_display)}",
                      (10, 35), font_scale=1.0,
                      color=COL_GREEN if not is_bg else COL_RED)
        put_text_safe(frame, f"Word  : {safe(current_word) or '--'}",
                      (10, 75), font_scale=0.8, color=COL_YELLOW)
        return frame

    h, w = frame.shape[:2]
    # Single BGR→RGB→PIL conversion for this frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    dark  = (30, 30, 30)
    char_color = (0, 220, 0) if not is_bg else (220, 0, 0)   # RGB
    yellow_rgb = (200, 200, 0)
    white_rgb  = (255, 255, 255)

    font_big   = _FONT_BIG
    font_small = _FONT_SMALL or _FONT_BIG

    # ── Predicted letter ──────────────────────────────────────────────────
    letter_text = f"Letter: {char_display}"
    bbox = draw.textbbox((10, 8), letter_text, font=font_big)
    draw.rectangle([bbox[0]-4, bbox[1]-4, bbox[2]+4, bbox[3]+4], fill=dark)
    draw.text((10, 8), letter_text, font=font_big, fill=char_color)

    # ── Current word ──────────────────────────────────────────────────────
    word_text = f"Word: {current_word if current_word else '--'}"
    bbox2 = draw.textbbox((10, 52), word_text, font=font_small)
    draw.rectangle([bbox2[0]-4, bbox2[1]-4, bbox2[2]+4, bbox2[3]+4], fill=dark)
    draw.text((10, 52), word_text, font=font_small, fill=yellow_rgb)

    # ── Finalized word flash (centred banner) ─────────────────────────────
    if finalized:
        fin_text = f"  Word: {finalized}  "
        bbox3 = draw.textbbox((0, 0), fin_text, font=font_big)
        tw = bbox3[2] - bbox3[0]
        x = max(0, (w - tw) // 2)
        y = h // 2 - 30
        draw.rectangle([x - 8, y - 8, x + tw + 8, y + 50], fill=(0, 130, 0))
        draw.text((x, y), fin_text, font=font_big, fill=white_rgb)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── Majority voting ───────────────────────────────────────────────────────────

class MajorityVoter:
    def __init__(self, window: int = VOTE_WINDOW):
        self.window = window
        self.buffer  = deque(maxlen=window)

    def update(self, prediction: int) -> int:
        self.buffer.append(prediction)
        if len(self.buffer) < self.window // 2:
            return prediction
        return Counter(self.buffer).most_common(1)[0][0]

    def reset(self):
        self.buffer.clear()


# ── Word formation ────────────────────────────────────────────────────────────

class WordFormer:
    """Accumulates stable letter predictions into a running word."""

    def __init__(
        self,
        add_delay:  float = ADD_LETTER_DELAY,
        gap_delay:  float = WORD_GAP_DELAY,
    ):
        self.add_delay  = add_delay
        self.gap_delay  = gap_delay
        self.current_word: list = []
        self.last_letter: str   = ''
        self.letter_stable_since: float = 0.0
        self.background_since:    float = 0.0
        self.background_active:   bool  = True
        self.word_history: list   = []

    def update(self, letter: str, is_bg: bool, now: float) -> tuple:
        """
        Update state with the latest smoothed prediction.
        Returns (current_word_str, just_added, just_finalized_word).
        """
        just_added     = False
        finalized_word = None

        if is_bg:
            # Background — start gap timer
            if not self.background_active:
                self.background_since  = now
                self.background_active = True
            elif now - self.background_since >= self.gap_delay:
                # Long gap → finalize current word
                if self.current_word:
                    finalized_word = ''.join(self.current_word)
                    self.word_history.append(finalized_word)
                    self.current_word = []
                self.last_letter        = ''
                self.letter_stable_since = 0.0
            return ''.join(self.current_word), just_added, finalized_word

        # Not background
        self.background_active = False

        if letter != self.last_letter:
            self.last_letter         = letter
            self.letter_stable_since = now
        else:
            # Same letter for long enough → add it
            if now - self.letter_stable_since >= self.add_delay:
                self.current_word.append(letter)
                self.letter_stable_since = now  # reset timer (don't double-add)
                just_added = True

        return ''.join(self.current_word), just_added, finalized_word


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(
    camera_idx:  int   = 0,
    threshold:   float = CONFIDENCE_THRES,
    window:      int   = VOTE_WINDOW,
    show_window: bool  = True,
):
    setup_utf8_console()

    model        = load_model()
    idx_to_folder = load_class_map()
    word_list    = load_word_list(os.path.normpath(README_PATH))
    voter        = MajorityVoter(window)
    word_former  = WordFormer()

    # Warm-up inference (avoids first-frame latency spike)
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("[OK] Model warmed up.")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_idx}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 60)
    print("  Tamil Sign Recognition — Real-Time Inference")
    print("  Press  Q  or  ESC  to quit")
    print("  Press  C  to clear the current word")
    print("=" * 60 + "\n")

    fps_timer      = time.time()
    frame_count    = 0
    fps            = 0.0
    last_char      = ''
    last_conf      = 0.0
    last_print_time = 0.0          # for 1-second terminal refresh
    flash_word      = ''           # word to show in banner on screen
    flash_until     = 0.0          # keep banner visible until this time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed - retrying...")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)   # mirror for natural interaction
        now   = time.time()

        # ── Inference ──────────────────────────────────────────────────────
        blob   = preprocess_frame(frame)
        preds  = model.predict(blob, verbose=0)[0]
        raw_idx  = int(np.argmax(preds))
        conf     = float(preds[raw_idx])

        if conf >= threshold:
            smoothed_idx = voter.update(raw_idx)
        else:
            smoothed_idx = voter.update(raw_idx)   # still feed—just low conf
            # treat as background if below threshold
            smoothed_idx = raw_idx  # keep original; flag below

        folder_name = idx_to_folder.get(smoothed_idx, 'Background')
        is_bg       = (folder_name == 'Background') or (conf < threshold)
        tamil_char  = 'Background' if is_bg else IDX_TO_CHAR.get(smoothed_idx, '?')

        # ── Word formation ─────────────────────────────────────────────────
        current_word, just_added, finalized = word_former.update(
            tamil_char, is_bg, now
        )

        # ── Terminal + log output ──────────────────────────────────────────
        # Always print current letter once per second so terminal stays live
        if now - last_print_time >= 1.0:
            last_print_time = now
            if not is_bg:
                log(f"[Letter] {tamil_char:<6}  conf: {conf:.0%}  "
                    f"| Word: {current_word if current_word else '--'}")
            else:
                log(f"[Letter] --      conf: {conf:.0%}  "
                    f"| Word: {current_word if current_word else '--'}")

        if just_added:
            log(f"  >> Letter Added : {tamil_char}")
            log(f"  >> Current Word : {current_word}")

        if finalized:
            log(f"\n{'='*40}")
            log(f"  WORD FORMED: {finalized}")
            log(f"{'='*40}\n")
            flash_word  = finalized
            flash_until = now + 3.0    # show banner on screen for 3 seconds

        # ── FPS counter ────────────────────────────────────────────────────
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps        = frame_count / (time.time() - fps_timer)
            frame_count = 0
            fps_timer   = time.time()

        # ── GUI overlay ────────────────────────────────────────────────────
        if show_window:
            draw_roi_box(frame)

            h, w = frame.shape[:2]

            # Confidence bar
            bar_w = int(conf * 200)
            bar_colour = COL_GREEN if conf >= threshold else COL_RED
            cv2.rectangle(frame, (10, h - 30), (210, h - 10), COL_DARK, -1)
            cv2.rectangle(frame, (10, h - 30), (10 + bar_w, h - 10),
                          bar_colour, -1)
            cv2.putText(frame, f"{conf:.0%}", (215, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1)

            # FPS + controls (ASCII — cv2 is fine)
            put_text_safe(frame, f"FPS: {fps:.1f}",
                          (w - 120, 35), font_scale=0.6, color=COL_BLUE,
                          bg=False)
            cv2.putText(frame, "Q/ESC=Quit  C=Clear",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, COL_WHITE, 1, cv2.LINE_AA)

            # Tamil text overlays (letter + word + finalized banner)
            char_display = tamil_char if tamil_char != 'Background' else 'BG'
            banner = flash_word if now < flash_until else None
            frame = apply_tamil_overlays(
                frame, char_display, current_word, is_bg,
                finalized=banner
            )

            cv2.imshow("Tamil Sign Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):        # q or ESC
            break
        elif key == ord('c'):
            word_former.current_word = []
            voter.reset()
            print("[INFO] Word cleared.")

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    # ── Session summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Session Summary")
    print("=" * 60)
    if word_former.word_history:
        print("  Words formed this session:")
        for w in word_former.word_history:
            print(f"    {w}")
    else:
        print("  No complete words formed.")
    print("[DONE] Session ended.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Real-time Tamil Sign Recognition'
    )
    parser.add_argument('--camera',     type=int,   default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--threshold',  type=float, default=CONFIDENCE_THRES,
                        help=f'Confidence threshold (default: {CONFIDENCE_THRES})')
    parser.add_argument('--window',     type=int,   default=VOTE_WINDOW,
                        help=f'Majority-vote window size (default: {VOTE_WINDOW})')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable OpenCV window (terminal output only)')
    args = parser.parse_args()

    run(
        camera_idx  = args.camera,
        threshold   = args.threshold,
        window      = args.window,
        show_window = not args.no_display,
    )
