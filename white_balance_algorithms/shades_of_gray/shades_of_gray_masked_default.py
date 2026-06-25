import numpy as np
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class ShadesOfGrayMaskedDefault(WhiteBalanceAlgorithm):
    def __init__(self):
        self.p = 6
        self.mask = 0.99

    def _estimate(self, data, process_masked=False):
        pixels = self._get_pixels(data.get_raw_image(), data, process_masked)
        valid = pixels[np.all(pixels < self.mask, axis=1)]
        src = valid if valid.size > 0 else pixels
        pth_l_norm = np.power(np.mean(np.power(src, self.p), axis=0), 1.0 / self.p)
        b_avg, g_avg, r_avg = pth_l_norm
        if g_avg == 0:
            g_avg = 1e-6
        return {
            "single_illuminant": (r_avg / g_avg, b_avg / g_avg),
            "multi_illuminants": None,
            "illuminant_map": None,
            "estimated_srgb_image": None,
        }
