import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# KONFIGURASI PATH
# ==========================================================
BASE_DIR = r"C:\Users\ASUS ROG Strix\OneDrive\Dokumen\TUBESPCD"
CSV_PATH = os.path.join(BASE_DIR, "results", "evaluation_results.csv")
PLOT_DIR = os.path.join(BASE_DIR, "results", "plots")

# Buat folder plot jika belum ada
os.makedirs(PLOT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================
print("Membaca CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

methods = df['Method'].unique()

# ==========================================================
# PLOT MSE
# ==========================================================
plt.figure(figsize=(12, 6))
for method in methods:
    subset = df[df['Method'] == method]
    plt.plot(subset['MSE'].values, label=method)

plt.title("Perbandingan MSE Antar Metode")
plt.xlabel("Index Gambar")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)

mse_output = os.path.join(PLOT_DIR, "mse_plot.png")
plt.savefig(mse_output)
plt.show()

print("Grafik MSE tersimpan di:", mse_output)

# ==========================================================
# PLOT SSIM
# ==========================================================
plt.figure(figsize=(12, 6))
for method in methods:
    subset = df[df['Method'] == method]
    plt.plot(subset['SSIM'].values, label=method)

plt.title("Perbandingan SSIM Antar Metode")
plt.xlabel("Index Gambar")
plt.ylabel("SSIM")
plt.legend()
plt.grid(True)

ssim_output = os.path.join(PLOT_DIR, "ssim_plot.png")
plt.savefig(ssim_output)
plt.show()

print("Grafik SSIM tersimpan di:", ssim_output)

print("\n=== Semua grafik berhasil dibuat! ===")
