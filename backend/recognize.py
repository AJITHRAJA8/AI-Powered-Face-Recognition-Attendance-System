# ============================================================
#   recognize.py  (UPGRADED – LIGHTING ROBUST + MULTI-FACE)
#   • Uses encodings from trained_faces.pkl
#   • Normalizes embeddings for stable distance comparison
#   • Histogram equalization (LAB) for lighting-invariant faces
#   • Multi-camera support: USB / IP / RTSP
#   • Simple liveness heuristic (movement-based)
#   • Unauthorized hint (unknown face or low liveness)
#   • Sends JSON events to backend /api/log-face-event
#   • Draws debug overlays in OpenCV windows
#   • NEW: sends events for ALL faces in the frame every interval
# ============================================================

import cv2
import numpy as np
import face_recognition
import pickle
import argparse
import threading
import requests
import time
from datetime import datetime

# ============================================================
#   CONFIG
# ============================================================

PICKLE_FILE = "trained_faces.pkl"
BACKEND_URL = "http://127.0.0.1:5000/api/log-face-event"  # Flask backend from app.py
EVENT_INTERVAL_SEC = 4  # throttle sending events per camera (seconds)


# ============================================================
#   HELPER: EMBEDDING NORMALIZATION
# ============================================================

def _normalize_vec(v):
    """
    L2-normalize a vector for more stable cosine-like distance.
    """
    v = np.array(v, dtype="float32")
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


# ============================================================
#   LOAD ENCODINGS
# ============================================================

print(f"🔄 Loading face encodings from {PICKLE_FILE} ...")

with open(PICKLE_FILE, "rb") as f:
    staff_data = pickle.load(f)

KNOWN_ENCODINGS = []
KNOWN_UIDS = []
KNOWN_NAMES = []

for uid, info in staff_data.items():
    for enc in info.get("encodings", []):
        KNOWN_ENCODINGS.append(enc)
        KNOWN_UIDS.append(uid)
        KNOWN_NAMES.append(info.get("name", uid))

# Normalize all known encodings for consistent distance behavior
KNOWN_ENCODINGS = [_normalize_vec(e) for e in KNOWN_ENCODINGS]

print(f"✅ Loaded {len(KNOWN_ENCODINGS)} encodings for {len(staff_data)} staff.")


# ============================================================
#   BACKEND EVENT SENDER
# ============================================================

def send_event(uid: str,
               name: str,
               confidence: float,
               event_type: str,
               camera_id: str,
               liveness_score: float,
               unauthorized_hint: bool) -> None:
    """
    Send event to backend HTTP API (Flask app.py).
    Backend decides:
      • lateness (on-time / late / unauthorized)
      • writes to Firebase (face_events, alerts, etc.)
    """
    payload = {
        "uid": uid,  # always send uid; backend will decide unauthorized if no permission
        "name": name,
        "confidence": round(float(confidence), 2),
        "event_type": event_type,   # "entry" / "exit" or "in"/"out"
        "camera_id": camera_id,     # "CAM_1", "CAM_2", ...
        "liveness": float(liveness_score),
        "unauthorized_hint": bool(unauthorized_hint),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=1.5)
        if r.status_code != 200:
            print(f"⚠️ Backend responded with {r.status_code}: {r.text[:120]}")
    except Exception as e:
        print(f"❌ Failed to send event: {e}")


# ============================================================
#   LIVENESS DETECTION (simple heuristic)
# ============================================================

def compute_liveness(prev_face, curr_face) -> float:
    """
    Simple liveness check: compares difference between
    consecutive face crops. NOT production-grade anti-spoofing,
    but helps against static photo attacks.
    Returns score between 0..1 (higher = more movement).
    """
    if prev_face is None or curr_face is None:
        return 1.0

    try:
        prev_resized = cv2.resize(prev_face, (64, 64))
        curr_resized = cv2.resize(curr_face, (64, 64))
    except Exception:
        return 1.0

    diff = cv2.absdiff(prev_resized, curr_resized)
    score = diff.mean() / 255.0  # 0..1
    score = min(1.0, max(0.0, score * 3.0))  # scale up a bit

    return float(score)


# ============================================================
#   FACE MATCHING
# ============================================================

def match_face(encoding, threshold: float = 0.52):
    """
    Compare a face encoding against known encodings.
    Returns:
        (matched: bool, uid: str, name: str, confidence: float)

    threshold:
        Higher (~0.52) to be slightly more tolerant to lighting
        variations when using normalized embeddings.
    """
    if not KNOWN_ENCODINGS:
        return False, "unknown", "Unknown", 0.0

    # Normalize input encoding same as reference encodings
    encoding = _normalize_vec(encoding)

    dists = face_recognition.face_distance(KNOWN_ENCODINGS, encoding)
    idx = int(np.argmin(dists))
    dist = float(dists[idx])

    # Simple "confidence" heuristic
    confidence = max(0.0, (1.0 - dist) * 100.0)

    if dist < threshold:
        return True, KNOWN_UIDS[idx], KNOWN_NAMES[idx], confidence

    return False, "unknown", "Unknown", confidence


# ============================================================
#   CAMERA LOOP WORKER (MULTI-FACE EVENTS)
# ============================================================

def camera_worker(src, event_type: str, camera_id: str):
    """
    src: camera index or RTSP/HTTP URL
    event_type: "entry", "exit", or similar (this goes to backend as event_type)
    camera_id: label for debug + backend
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"❌ Cannot open camera source: {src}")
        return

    last_face = None
    last_sent = 0.0

    print(f"📹 Camera started: {camera_id} (source={src})  event_type={event_type}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Frame read failed on {camera_id}, stopping.")
            break

        # Half-size for speed
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ====================================================
        #   LIGHTING NORMALIZATION (LAB HISTOGRAM EQUALIZATION)
        # ====================================================
        try:
            lab = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)  # equalize luminance channel
            lab = cv2.merge((l, a, b))
            rgb_small = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception:
            # If anything fails, just proceed with original rgb_small
            pass

        # Face detection & encoding on lighting-normalized frame
        boxes = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, boxes)

        now = time.time()

        # Collect all face events for this frame
        frame_events = []  # list of dicts: one per face

        for (top, right, bottom, left), enc in zip(boxes, encs):
            face_crop = small[top:bottom, left:right]
            live_score = compute_liveness(last_face, face_crop)
            last_face = face_crop.copy()

            matched, uid, name, conf = match_face(enc)
            unauthorized_hint = (not matched) or (live_score <= 0.30)

            # Store event for this face (we'll send all together below)
            frame_events.append(
                (uid, name, conf, live_score, unauthorized_hint)
            )

            # Draw overlay
            color = (0, 255, 0)
            label = f"{name} {conf:.1f}% L:{live_score:.2f}"

            if unauthorized_hint:
                color = (0, 0, 255)
                label = f"UNAUTH? {conf:.1f}% L:{live_score:.2f}"

            scale = 2  # we resized by 0.5, so multiply by 2
            cv2.rectangle(
                frame,
                (left * scale, top * scale),
                (right * scale, bottom * scale),
                color,
                2,
            )
            cv2.putText(
                frame,
                label,
                (left * scale, max(15, top * scale - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # ✅ NEW: Send events for ALL faces in this frame every interval
        if frame_events and (now - last_sent > EVENT_INTERVAL_SEC):
            for uid, name, conf, live_score, unauthorized_hint in frame_events:
                send_event(uid, name, conf, event_type, camera_id, live_score, unauthorized_hint)
            last_sent = now

        cv2.imshow(camera_id, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"🛑 Stopping {camera_id}")
            break

    cap.release()
    cv2.destroyWindow(camera_id)


# ============================================================
#   PARSE MULTI-SOURCES
# ============================================================

def parse_sources(src_string: str):
    """
    Convert "0,1,rtsp://..." → [0, 1, "rtsp://..."]
    """
    result = []
    parts = src_string.split(",")

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.isdigit():
            result.append(int(p))
        else:
            result.append(p)

    return result


# ============================================================
#   MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        default="0",
        help="Comma separated camera list, e.g. 0,1,rtsp://user:pass@ip/stream",
    )
    parser.add_argument(
        "--event_type",
        default="entry",
        help="Logical event type: entry / exit (stored as event_type in backend).",
    )

    args = parser.parse_args()
    sources = parse_sources(args.sources)
    event_type = (args.event_type or "entry").lower().strip()

    threads = []
    for i, src in enumerate(sources):
        cam_id = f"CAM_{i + 1}"
        t = threading.Thread(
            target=camera_worker,
            args=(src, event_type, cam_id),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print("🚀 Multi-camera recognition running. Press CTRL+C or 'q' in a window to exit.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("🛑 Stopping all cameras...")
        # threads are daemon, they exit with main
