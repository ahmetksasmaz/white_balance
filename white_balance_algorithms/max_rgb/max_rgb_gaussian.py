import cv2 as cv
import numpy as np
from datasets.data import Data
from white_balance_algorithms.single_illuminant_estimation import SingleIlluminantEstimationAlgorithm

class MaxRGBGaussian(SingleIlluminantEstimationAlgorithm):
    def __init__(self, kernel_size=5, sigma=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def _estimate(self, data):
        image = data.get_raw_image()  # image is normalized to 0-1
        # Apply Gaussian blur to the image
        blurred = cv.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        # Compute the maximum color value for each channel
        b_max = np.max(blurred[:, :, 0])
        g_max = np.max(blurred[:, :, 1])
        r_max = np.max(blurred[:, :, 2])
        # Avoid division by zero
        if g_max == 0:
            g_max = 1e-6
        r_g = r_max / g_max
        b_g = b_max / g_max
        return (r_g, b_g)