import numpy as np
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class GrayWorldNaive(WhiteBalanceAlgorithm):
    def _estimate(self, data, process_masked=False):
        pixels = self._get_pixels(data.get_raw_image(), data, process_masked)
        b_avg, g_avg, r_avg = pixels.mean(axis=0)
        if g_avg == 0:
            g_avg = 1e-6
        return {
            "single_illuminant": (r_avg / g_avg, b_avg / g_avg),
            "multi_illuminants": None,
            "illuminant_map": None,
            "estimated_srgb_image": None,
        }
