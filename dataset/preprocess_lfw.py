import os
import cv2
from pathlib import Path
from tqdm import tqdm

# ===========================
# KONFIGURASI
# ===========================
INPUT_DIR = "datasetLFW/lfw-deepfunneled/lfw-deepfunneled"
OUTPUT_DIR = "datasetLFW/lfw_preprocessed"
TARGET_SIZE = (128, 128)               

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Haar Cascade untuk deteksi wajah
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(haar_path)

# ===========================
# FUNGSI PROSES CITRA
# ===========================
def process_image(infile, outfile):
    """Proses satu gambar: deteksi wajah → crop → grayscale → resize"""
    img = cv2.imread(infile)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # deteksi wajah
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        # fallback jika wajah tidak terdeteksi: center crop square
        h, w = gray.shape
        m = min(h, w)
        crop = gray[(h-m)//2:(h+m)//2, (w-m)//2:(w+m)//2]
    else:
        # gunakan wajah pertama yang terdeteksi
        x, y, w, h = faces[0]

        # berikan margin agar crop lebih baik
        margin = int(0.25 * max(w, h))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(gray.shape[1], x + w + margin)
        y1 = min(gray.shape[0], y + h + margin)

        crop = gray[y0:y1, x0:x1]

    # resize citra hasil crop
    resized = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    cv2.imwrite(outfile, resized)
    return True

# ===========================
# LOOP SEMUA FILE
# ===========================
print("Memulai preprocessing dataset LFW...")

for root, _, files in os.walk(INPUT_DIR):
    for fname in tqdm(files):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            inpath = os.path.join(root, fname)

            # buat struktur folder output yang sama
            rel_folder = os.path.relpath(root, INPUT_DIR)
            outdir = os.path.join(OUTPUT_DIR, rel_folder)
            os.makedirs(outdir, exist_ok=True)

            outpath = os.path.join(outdir, fname)

            process_image(inpath, outpath)

print("Selesai! Hasil tersimpan di folder:", OUTPUT_DIR)
