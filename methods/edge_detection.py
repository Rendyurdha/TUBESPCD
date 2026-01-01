import cv2
import numpy as np
from scipy import ndimage

# ================================
# SOBEL
# ================================
def sobel_edge(img):
    """Deteksi tepi menggunakan Sobel"""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sobely, sobelx)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    return mag

# ================================
# PREWITT
# ================================
def prewitt_edge(img):
    """Deteksi tepi menggunakan Prewitt"""
    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])

    kernely = kernelx.T

    gx = ndimage.convolve(img.astype(float), kernelx)
    gy = ndimage.convolve(img.astype(float), kernely)

    mag = np.hypot(gx, gy)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    return mag

# ================================
# CANNY
# ================================
def canny_edge(img):
    """Deteksi tepi menggunakan Canny"""
    edges = cv2.Canny(img, 50, 150)
    return edges
