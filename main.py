import os
import cv2
from tqdm import tqdm

# Import fungsi dari folder methods
from methods.edge_detection import sobel_edge, prewitt_edge, canny_edge
from methods.frequency_filter import fft_lowpass, fft_highpass, fft_bandstop

# ==========================
# KONFIGURASI
# ==========================
INPUT_DIR = "datasetLFW/lfw_preprocessed"
OUT_EDGES = "results/edges"
OUT_FREQ = "results/freq"

os.makedirs(OUT_EDGES, exist_ok=True)
os.makedirs(OUT_FREQ, exist_ok=True)

# ==========================
# MAIN PROCESS LOOP
# ==========================
def process_all_images():

    print("\n=== Memulai pemrosesan EDGE DETECTION ===\n")

    for root, _, files in os.walk(INPUT_DIR):
        for fname in tqdm(files):
            
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                inpath = os.path.join(root, fname)
                img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print("Gambar rusak / tidak bisa dibaca:", inpath)
                    continue

                # buat folder output
                rel = os.path.relpath(root, INPUT_DIR)
                out_edge_folder = os.path.join(OUT_EDGES, rel)
                os.makedirs(out_edge_folder, exist_ok=True)

                # pecah nama + ekstensi
                base, ext = os.path.splitext(fname)

                # EDGE DETECTION
                sobel_img = sobel_edge(img)
                prewitt_img = prewitt_edge(img)
                canny_img = canny_edge(img)

                cv2.imwrite(os.path.join(out_edge_folder, f"{base}_sobel{ext}"), sobel_img)
                cv2.imwrite(os.path.join(out_edge_folder, f"{base}_prewitt{ext}"), prewitt_img)
                cv2.imwrite(os.path.join(out_edge_folder, f"{base}_canny{ext}"), canny_img)

    print("\n=== Memulai pemrosesan FREQUENCY FILTERING ===\n")

    for root, _, files in os.walk(INPUT_DIR):
        for fname in tqdm(files):

            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                inpath = os.path.join(root, fname)
                img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print("Gambar rusak / tidak bisa dibaca:", inpath)
                    continue

                rel = os.path.relpath(root, INPUT_DIR)
                out_freq_folder = os.path.join(OUT_FREQ, rel)
                os.makedirs(out_freq_folder, exist_ok=True)

                base, ext = os.path.splitext(fname)

                # FFT FILTERING
                low = fft_lowpass(img, cutoff=0.05)
                high = fft_highpass(img, cutoff=0.05)
                band = fft_bandstop(img, cutoff_low=0.02, cutoff_high=0.08)

                cv2.imwrite(os.path.join(out_freq_folder, f"{base}_lowpass{ext}"), low)
                cv2.imwrite(os.path.join(out_freq_folder, f"{base}_highpass{ext}"), high)
                cv2.imwrite(os.path.join(out_freq_folder, f"{base}_bandstop{ext}"), band)

    print("\n=== Semua pemrosesan selesai! ===\n")


if __name__ == "__main__":
    process_all_images()
