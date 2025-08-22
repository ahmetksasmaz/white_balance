import cv2 as cv
import numpy as np

def gamma_correction(image, gamma = 2.2):
    corrected_image = image.copy()
    corrected_image = np.power(corrected_image, 1.0 / gamma)
    return corrected_image

def histogram(image):
    hist = cv.calcHist([image], [0], None, [256], [0.0, 1.0])
    return hist

def pdf(image):
    hist = histogram(image)
    pdf = hist / np.sum(hist)
    return pdf

def log_chrominance(image):
    b,g,r = cv.split(image)
    log_u = np.log1p(g) - np.log1p(r+0.00000001)
    log_v = np.log1p(g) - np.log1p(b+0.00000001)
    return log_u, log_v

def prepare_display(image, correct_gamma=False):
    if correct_gamma:
        display_image = gamma_correction(image.copy())
    else:
        display_image = image.copy()
    display_image *= 255.0
    display_image = np.clip(display_image, 0, 255).astype(np.uint8)
    return display_image

def lab_to_srgb(illuminant):
    xyz = lab_to_xyz(illuminant)
    return xyz_to_srgb(xyz)

def srgb_to_lab(illuminant):
    xyz = srgb_to_xyz(illuminant)
    return xyz_to_lab(xyz)

def srgb_to_xyz(illuminant):
    r, g, b = illuminant
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return (x, y, z)

def lab_to_xyz(illuminant):
    l, a, b = illuminant
    y = (l + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    x = 95.047 * (x ** 3) if x > 0.206893 else (x - 16/116) / 7.787
    y = 100.000 * (y ** 3) if y > 0.206893 else (y - 16/116) / 7.787
    z = 108.883 * (z ** 3) if z > 0.206893 else (z - 16/116) / 7.787

    return (x, y, z)

def xyz_to_lab(illuminant):
    x, y, z = illuminant
    x = x / 95.047
    y = y / 100.000
    z = z / 108.883

    x = x ** (1/3) if x > 0.008856 else (x * 7.787 + 16/116)
    y = y ** (1/3) if y > 0.008856 else (y * 7.787 + 16/116)
    z = z ** (1/3) if z > 0.008856 else (z * 7.787 + 16/116)

    l = max(0, min(100, (116 * y - 16)))
    a = max(-128, min(127, (x - y) * 500))
    b = max(-128, min(127, (y - z) * 200))
    return (l, a, b)

def xyz_to_srgb(illuminant):
    x, y, z = illuminant
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
    return (r, g, b)