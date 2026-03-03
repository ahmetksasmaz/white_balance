import cv2 as cv
import numpy as np
from datasets.data import Data
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class ShadesOfGrayMaskedP3(WhiteBalanceAlgorithm):
    def __init__(self):
        super().__init__()
        self.p = 3
        self.mask = 0.99
    
    def _estimate(self, data, process_masked=False):
        image = data.get_raw_image()  # image is normalized to 0-1
        pixels = self._get_pixels(image, data, process_masked)  # (N, 3)
        # Minkowski p-norm per channel
        valid_pixels = pixels[np.all(pixels < self.mask, axis=1)]
        if valid_pixels.size == 0:
            # Fallback to mean of all pixels if all are saturated (unlikely but safe)
            pth_l_norm = np.power(np.mean(np.power(pixels, self.p), axis=0), 1.0 / self.p)
        else:
            pth_l_norm = np.power(np.mean(np.power(valid_pixels, self.p), axis=0), 1.0 / self.p)
        b_avg, g_avg, r_avg = pth_l_norm
        # Avoid division by zero
        if g_avg == 0:
            g_avg = 1e-6
        r_g = r_avg / g_avg
        b_g = b_avg / g_avg
        return {
            "single_illuminant": (r_g, b_g),
            "multi_illuminants": None,
            "illuminant_map": None,
            "estimated_srgb_image": None
        }