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