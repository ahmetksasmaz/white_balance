import cv2 as cv
import numpy as np
from datasets.data import Data
from white_balance_algorithms.single_illuminant_estimation import SingleIlluminantEstimationAlgorithm

class MaxRGBNaive(SingleIlluminantEstimationAlgorithm):
    def __init__(self):
        super().__init__()
    
    def _estimate(self, data):
        image = data.get_raw_image()  # image is normalized to 0-1
        # Compute the maximum color value for each channel
        r_max = np.max(image[:, :, 0])
        g_max = np.max(image[:, :, 1])
        b_max = np.max(image[:, :, 2])
        # Avoid division by zero
        if g_max == 0:
            g_max = 1e-6
        r_g = r_max / g_max
        b_g = b_max / g_max
        return (r_g, b_g)