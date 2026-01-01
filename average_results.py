import os
import pandas as pd

# ==========================================================
# PATH
# ==========================================================
BASE_DIR = r"C:\Users\ASUS ROG Strix\OneDrive\Dokumen\TUBESPCD"
CSV_PATH = os.path.join(BASE_DIR, "results", "evaluation_results.csv")

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(CSV_PATH)

# ==========================================================
# RATA-RATA PER METODE
# ==========================================================
avg_per_method = (
    df.groupby("Method")[["MSE", "SSIM"]]
    .mean()
    .reset_index()
)

print("\n=== Rata-rata MSE dan SSIM per Metode ===")
print(avg_per_method)

# ==========================================================
# KELOMPOK METODE
# ==========================================================
edge_methods = ["sobel", "prewitt", "canny"]
freq_methods = ["lowpass", "highpass", "bandstop"]

df_edges = df[df["Method"].isin(edge_methods)]
df_freq = df[df["Method"].isin(freq_methods)]

avg_edges = df_edges[["MSE", "SSIM"]].mean()
avg_freq = df_freq[["MSE", "SSIM"]].mean()

print("\n=== Rata-rata KELOMPOK EDGE ===")
print(avg_edges)

print("\n=== Rata-rata KELOMPOK FREQUENCY ===")
print(avg_freq)

# ==========================================================
# SIMPAN KE CSV (OPSIONAL)
# ==========================================================
output_path = os.path.join(BASE_DIR, "results", "average_summary.csv")

summary = pd.DataFrame({
    "Group": ["Edge-based", "Frequency-based"],
    "Avg_MSE": [avg_edges["MSE"], avg_freq["MSE"]],
    "Avg_SSIM": [avg_edges["SSIM"], avg_freq["SSIM"]],
})

summary.to_csv(output_path, index=False)
print("\nFile ringkasan disimpan di:", output_path)
