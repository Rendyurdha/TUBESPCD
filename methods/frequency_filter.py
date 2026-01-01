import numpy as np
import numpy.fft as fft

# ============================================================
# Fourier Transform Based Frequency Filtering
# ============================================================

def fft_lowpass(img, cutoff=0.05):
    """
    Low-pass filter menggunakan FFT.
    cutoff: proporsi radius area frekuensi rendah yang dipertahankan.
    """
    f = fft.fft2(img)
    fshift = fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    radius = int(min(rows, cols) * cutoff)

    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1

    fshift_filtered = fshift * mask
    f_ishift = fft.ifftshift(fshift_filtered)
    img_back = fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = (img_back / img_back.max() * 255).astype(np.uint8)
    return img_back


def fft_highpass(img, cutoff=0.05):
    """
    High-pass filter menggunakan FFT.
    cutoff: proporsi radius frekuensi rendah yang dihilangkan.
    """
    f = fft.fft2(img)
    fshift = fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    radius = int(min(rows, cols) * cutoff)

    mask = np.ones((rows, cols), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0

    fshift_filtered = fshift * mask
    f_ishift = fft.ifftshift(fshift_filtered)
    img_back = fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = (img_back / img_back.max() * 255).astype(np.uint8)
    return img_back


def fft_bandstop(img, cutoff_low=0.02, cutoff_high=0.08):
    """
    Band-stop filter (menhilangkan frekuensi menengah).
    cutoff_low < cutoff_high
    """
    f = fft.fft2(img)
    fshift = fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    r_low = int(min(rows, cols) * cutoff_low)
    r_high = int(min(rows, cols) * cutoff_high)

    mask = np.ones((rows, cols), np.uint8)

    # nol-kan frekuensi antara r_low dan r_high
    mask[crow - r_high : crow + r_high,
         ccol - r_high : ccol + r_high] = 0

    # tapi biarkan yang dalam r_low aktif lagi
    mask[crow - r_low : crow + r_low,
         ccol - r_low : ccol + r_low] = 1

    fshift_filtered = fshift * mask
    f_ishift = fft.ifftshift(fshift_filtered)
    img_back = fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = (img_back / img_back.max() * 255).astype(np.uint8)
    return img_back
