import cv2 as cv
import numpy as np
from datasets.data import Data
from white_balance_algorithms.single_illuminant_estimation import SingleIlluminantEstimationAlgorithm

class ShadesOfGrayDefault(SingleIlluminantEstimationAlgorithm):
    def __init__(self, p=6):
        super().__init__()
        self.p = p
    
    def _estimate(self, data):
        image = data.get_raw_image()  # image is normalized to 0-1
        # Compute the average color of the image
        pth_l_norm = np.power(np.mean(np.power(image, self.p), axis=(0, 1)), 1.0 / self.p)
        b_avg, g_avg, r_avg = pth_l_norm
        # Avoid division by zero
        if g_avg == 0:
            g_avg = 1e-6
        r_g = r_avg / g_avg
        b_g = b_avg / g_avg
        return (r_g, b_g)