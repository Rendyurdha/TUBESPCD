import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import csv

# ===========================================
# KONFIGURASI
# ===========================================
PREPRO_DIR = "datasetLFW/lfw_preprocessed"
EDGE_DIR = "results/edges"
FREQ_DIR = "results/freq"

OUT_CSV = "results/evaluation_results.csv"

# ===========================================
# FUNGSI PERHITUNGAN METRIK
# ===========================================
def mse(imgA, imgB):
    return np.mean((imgA - imgB) ** 2)

def compute_metrics(original, processed):
    m = mse(original, processed)
    s, _ = ssim(original, processed, full=True)
    return m, s

# ===========================================
# LOOP EVALUASI
# ===========================================
rows = []

for root, _, files in os.walk(PREPRO_DIR):

    for fname in files:
        if not fname.lower().endswith(".jpg"):
            continue

        original_path = os.path.join(root, fname)

        # baca citra asli
        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            continue

        # Rel path utk mencocokkan folder pada edges & freq
        rel = os.path.relpath(root, PREPRO_DIR)

        # PATH gambar hasil metode
        edge_folder = os.path.join(EDGE_DIR, rel)
        freq_folder = os.path.join(FREQ_DIR, rel)

        base = fname.replace(".jpg", "")

        paths = {
            "sobel": os.path.join(edge_folder, base + "_sobel.jpg"),
            "prewitt": os.path.join(edge_folder, base + "_prewitt.jpg"),
            "canny": os.path.join(edge_folder, base + "_canny.jpg"),
            "lowpass": os.path.join(freq_folder, base + "_lowpass.jpg"),
            "highpass": os.path.join(freq_folder, base + "_highpass.jpg"),
            "bandstop": os.path.join(freq_folder, base + "_bandstop.jpg"),
        }

        for method, p in paths.items():
            if os.path.exists(p):

                proc = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if proc is None:
                    continue

                m, s = compute_metrics(original, proc)

                rows.append([
                    base,
                    method,
                    m,
                    s
                ])

# ===========================================
# SIMPAN CSV
# ===========================================
os.makedirs("results", exist_ok=True)

with open(OUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Method", "MSE", "SSIM"])
    writer.writerows(rows)

print("\n=== Evaluasi selesai! ===")
print("File disimpan di:", OUT_CSV)
print("Total data:", len(rows))
