import cv2 as cv
import numpy as np
from datasets.data import Data
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class MaxRGBMedian(WhiteBalanceAlgorithm):
    def __init__(self):
        super().__init__()
        self.kernel_size = 5
    
    def _estimate(self, data, process_masked=False):
        image = data.get_raw_image()  # image is normalized to 0-1
        # Apply median blur to the image (spatial op on full image)
        blurred = cv.medianBlur(image.astype(np.float32), self.kernel_size)
        pixels = self._get_pixels(blurred, data, process_masked)  # (N, 3)
        b_max = np.max(pixels[:, 0])
        g_max = np.max(pixels[:, 1])
        r_max = np.max(pixels[:, 2])
        # Avoid division by zero
        if g_max == 0:
            g_max = 1e-6
        r_g = r_max / g_max
        b_g = b_max / g_max
        return {
            "single_illuminant": (r_g, b_g),
            "multi_illuminants": None,
            "illuminant_map": None,
            "estimated_srgb_image": None
        }