import cv2 as cv
import numpy as np

def gamma_correction(linear_image, gamma = 2.2):
    corrected_image = linear_image.copy()
    corrected_image = np.power(corrected_image, 1.0 / gamma)
    return corrected_image

def histogram(linear_image, bins=256):
    hist = cv.calcHist([linear_image], [0], None, [bins], [0.0, 1.0])
    return hist

def pdf(linear_image, bins=256):
    hist = histogram(linear_image, bins)
    pdf = hist / np.sum(hist)
    return pdf

def log_chrominance(linear_image):
    b,g,r = cv.split(linear_image)
    log_u = np.log1p(g) - np.log1p(r+0.00000001)
    log_v = np.log1p(g) - np.log1p(b+0.00000001)
    return log_u, log_v

def prepare_display(linear_image, correct_gamma=False):
    if correct_gamma:
        display_image = gamma_correction(linear_image)
    else:
        display_image = linear_image.copy()
    display_image *= 255.0
    display_image = np.clip(display_image, 0, 255).astype(np.uint8)
    return display_image