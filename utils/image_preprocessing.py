"""
Pre-process images before OCR to maximise text recognition accuracy.
Handles: deskew, denoising, binarisation, contrast enhancement.
"""

import io
import math
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Run the full preprocessing pipeline on a PIL image.
    Returns a cleaned, binarised grayscale image ready for OCR.
    """
    img = _ensure_rgb(image)
    img = _to_grayscale(img)
    img = _enhance_contrast(img)
    img = _denoise(img)
    img = _deskew(img)
    img = _binarise(img)
    img = _upscale_if_small(img)
    return img


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")


def _enhance_contrast(img: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(1.8)


def _denoise(img: Image.Image) -> Image.Image:
    # Median filter removes salt-and-pepper noise common in scans
    return img.filter(ImageFilter.MedianFilter(size=3))


def _deskew(img: Image.Image) -> Image.Image:
    """
    Estimate and correct rotation angle using projection profile analysis.
    Works well for skews up to ±15 degrees.
    """
    angle = _estimate_skew_angle(img)
    if abs(angle) < 0.3:
        return img
    # Expand=True prevents cropping of rotated content
    return img.rotate(angle, expand=True, fillcolor=255)


def _estimate_skew_angle(img: Image.Image) -> float:
    """
    Simple skew detection: binarise, compute horizontal projection profiles
    at candidate angles, pick the angle with maximum variance (sharpest lines).
    """
    try:
        arr = np.array(img.convert("L"))
        binary = (arr < 128).astype(np.uint8)

        best_angle = 0.0
        best_score = -1.0
        h, w = binary.shape

        for angle in np.arange(-10, 10.5, 0.5):
            rad = math.radians(angle)
            # Horizontal shear approximation
            rotated = np.zeros_like(binary)
            for row in range(h):
                shift = int(round(row * math.tan(rad)))
                if shift >= 0:
                    rotated[row, shift:] = binary[row, :w - shift]
                else:
                    rotated[row, :w + shift] = binary[row, -shift:]

            projection = rotated.sum(axis=1).astype(float)
            score = float(np.var(projection))
            if score > best_score:
                best_score = score
                best_angle = angle

        return best_angle
    except Exception:
        return 0.0


def _binarise(img: Image.Image) -> Image.Image:
    """
    Adaptive (Otsu-like) binarisation via PIL threshold.
    """
    arr = np.array(img)
    threshold = int(arr.mean() * 0.9)  # slightly below mean → cleaner text
    threshold = max(100, min(threshold, 200))
    return img.point(lambda p: 255 if p > threshold else 0, mode="L")


def _upscale_if_small(img: Image.Image, min_dpi_equivalent: int = 200) -> Image.Image:
    """
    OCR accuracy degrades sharply below ~200 DPI.
    If the image is very small (< 1200px wide), upscale 2×.
    """
    w, h = img.size
    if w < 1200:
        img = img.resize((w * 2, h * 2), Image.LANCZOS)
    return img


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()