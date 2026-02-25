import cv2 as cv
import numpy as np
from datasets.data import Data
from white_balance_algorithms.single_illuminant_estimation import SingleIlluminantEstimationAlgorithm

class GrayWorldNaive(SingleIlluminantEstimationAlgorithm):
    def __init__(self):
        super().__init__()
    
    def _estimate(self, data):
        image = data.get_raw_image()  # image is normalized to 0-1
        # Compute the average color of the image
        avg_color_per_channel = np.mean(image, axis=(0, 1))
        r_avg, g_avg, b_avg = avg_color_per_channel
        # Avoid division by zero
        if g_avg == 0:
            g_avg = 1e-6
        r_g = r_avg / g_avg
        b_g = b_avg / g_avg
        return (r_g, b_g)