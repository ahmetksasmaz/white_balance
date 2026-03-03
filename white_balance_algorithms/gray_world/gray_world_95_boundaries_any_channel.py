import cv2 as cv
import numpy as np
from datasets.data import Data
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class GrayWorld95BoundariesAnyChannel(WhiteBalanceAlgorithm):
    def __init__(self):
        super().__init__()
        self.lower_bound = 0.05
        self.upper_bound = 0.95
    
    def _estimate(self, data, process_masked=False):
        image = data.get_raw_image()  # image is normalized to 0-1
        pixels = self._get_pixels(image, data, process_masked)  # (N, 3)
        # Exclude pixels where all channels are below lower bound or all channels are above upper bound
        all_below = np.all(pixels < self.lower_bound, axis=-1)
        all_above = np.all(pixels > self.upper_bound, axis=-1)
        bounds_mask = ~(all_below | all_above)
        masked_pixels = pixels[bounds_mask]
        num_masked = masked_pixels.shape[0]
        if num_masked == 0:
            # Fallback: use mean over all valid pixels
            avg_color_per_channel = pixels.mean(axis=0)
        else:
            avg_color_per_channel = masked_pixels.mean(axis=0)
        b_avg, g_avg, r_avg = avg_color_per_channel
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