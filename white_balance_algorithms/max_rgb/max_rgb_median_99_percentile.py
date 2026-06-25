import cv2 as cv
import numpy as np
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class MaxRGBMedian99Percentile(WhiteBalanceAlgorithm):
    def __init__(self):
        self.kernel_size = 5
        self.percentile = 99

    def _estimate(self, data, process_masked=False):
        blurred = cv.medianBlur(data.get_raw_image().astype(np.float32), self.kernel_size)
        pixels = self._get_pixels(blurred, data, process_masked)
        b_max, g_max, r_max = np.percentile(pixels, self.percentile, axis=0)
        if g_max == 0:
            g_max = 1e-6
        return {
            "single_illuminant": (r_max / g_max, b_max / g_max),
            "multi_illuminants": None,
            "illuminant_map": None,
            "estimated_srgb_image": None,
        }
