import numpy as np
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class MaxRGB95Percentile(WhiteBalanceAlgorithm):
    def __init__(self):
        self.percentile = 95

    def _estimate(self, data, process_masked=False):
        pixels = self._get_pixels(data.get_raw_image(), data, process_masked)
        b_max, g_max, r_max = np.percentile(pixels, self.percentile, axis=0)
        if g_max == 0:
            g_max = 1e-6
        return {
            "single_illuminant": (r_max / g_max, b_max / g_max),
            "multi_illuminants": None,
            "illuminant_map": None,
            "estimated_srgb_image": None,
        }
