# ============================================================
#   train.py  (UPGRADED – LIGHTING ROBUST VERSION)
#   • Reads staff_details.csv  → staff_id, name, ...
#   • Reads images from staff_images/<staff_id>/
#   • Skips blurry / invalid images
#   • Performs lighting & noise augmentation per image
#       → simulates low-light, bright, contrast & noisy webcam frames
#   • Extracts multiple encodings per staff (original + augmented)
#   • Saves dictionary → trained_faces.pkl
#      {
#        "STAFF001": { "name": "Ajith", "encodings": [np.array(...), ...] },
#        ...
#      }
#   • Compatible with upgraded recognize.py
# ============================================================

import os 
import csv
import cv2 
import face_recognition
import pickle
import numpy as np
    
STAFF_CSV = "staff_details.csv"
IMAGE_DIR = "staff_images"
OUTPUT_FILE = "trained_faces.pkl"


# ============================================================
#   READ STAFF CSV
# ============================================================

def load_staff_csv():
    """
    Expect a CSV with headers like:
      staff_id, name, department, ...
    If staff_id not present, tries uid or id as fallback.
    """
    staff = {}

    if not os.path.exists(STAFF_CSV):
        print(f"❌ CSV not found: {STAFF_CSV}")
        return staff

    with open(STAFF_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("staff_id") or row.get("uid") or row.get("id")
            name = row.get("name") or uid
            if not uid:
                continue
            staff[uid] = {"name": name, "encodings": []}

    print(f"📘 Loaded {len(staff)} staff records from {STAFF_CSV}")
    return staff


# ============================================================
#   FACE QUALITY CHECK
# ============================================================

def is_blurry(image, threshold: float = 90.0) -> bool:
    """
    Simple blur detection using variance of Laplacian.
    Lower variance → blurrier image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


# ============================================================
#   LIGHTING & NOISE AUGMENTATION
# ============================================================

def augment_lighting(img):
    """
    Generate augmented variants of the input image to simulate:
      • darker / brighter lighting
      • contrast changes
      • webcam-type noise

    This allows training with ~10 base images while the model
    learns to recognize the same face across many light conditions.
    """
    augmented = []

    # Brightness variations (alpha = gain, beta = bias)
    for alpha in [0.6, 0.8, 1.2, 1.4]:
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=10)
        augmented.append(bright)

    # Contrast variations
    for alpha in [0.7, 1.3]:
        contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        augmented.append(contrast)

    # Add Gaussian-like noise (simulating low-quality webcam noise)
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    augmented.append(noisy)

    return augmented


# ============================================================
#   MAIN TRAINING FUNCTION
# ============================================================

def train_faces():
    staff = load_staff_csv()
    if not staff:
        print("❌ No staff info loaded. Check your CSV.") 
        return

    if not os.path.isdir(IMAGE_DIR):
        print(f"❌ Image directory not found: {IMAGE_DIR}")
        return

    for uid in staff.keys():
        folder = os.path.join(IMAGE_DIR, uid)
        if not os.path.isdir(folder):
            print(f"⚠️ Missing folder for {uid}: {folder}")
            continue

        print(f"\n🟦 Processing {uid}  ({staff[uid]['name']})")
        images_found = 0
        encodings_added = 0

        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            ext = file.lower().split(".")[-1]
            if ext not in ("jpg", "jpeg", "png", "bmp"):
                continue

            img = cv2.imread(path)
            if img is None:
                print(f"  ⚠️ Cannot read {file}, skipping.")
                continue

            images_found += 1

            # Skip very blurry originals (bad base data)
            if is_blurry(img):
                print(f"  ⚠️ Blurry image skipped: {file}")
                continue

            # Build list of images to encode:
            #   original + several lighting/noise augmented variants
            variants = [img] + augment_lighting(img)

            for variant in variants:
                # Resize for speed
                small = cv2.resize(variant, (0, 0), fx=0.5, fy=0.5)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                boxes = face_recognition.face_locations(rgb)
                if len(boxes) == 0:
                    # Only log for original image to avoid spam
                    continue

                encs = face_recognition.face_encodings(rgb, boxes)
                if not encs:
                    continue

                for enc in encs:
                    staff[uid]["encodings"].append(enc)
                    encodings_added += 1

        print(f"  ➜ Base images: {images_found}, total encodings stored (with augmentation): {encodings_added}")

    # Filter out staff with no encodings
    staff_filtered = {
        uid: info for uid, info in staff.items() if info["encodings"]
    }
    dropped = len(staff) - len(staff_filtered)
    if dropped > 0:
        print(f"\n⚠️ Dropped {dropped} staff with 0 valid encodings.")

    # Convert encodings to plain numpy arrays (picklable)
    for uid, info in staff_filtered.items():
        info["encodings"] = [np.array(e) for e in info["encodings"]]

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(staff_filtered, f)

    print("\n=======================================================")
    print(f"🎉 TRAINING COMPLETE! Saved {len(staff_filtered)} staff to {OUTPUT_FILE}")
    total_enc = sum(len(v["encodings"]) for v in staff_filtered.values())
    print(f"   Total encodings stored (original + augmented): {total_enc}")
    print("=======================================================")


# ============================================================
#   ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_faces()
