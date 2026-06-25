import numpy as np
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class GrayWorld95BoundariesAllChannels(WhiteBalanceAlgorithm):
    def __init__(self):
        self.lower_bound = 0.05
        self.upper_bound = 0.95

    def _estimate(self, data, process_masked=False):
        pixels = self._get_pixels(data.get_raw_image(), data, process_masked)
        bounds_mask = np.all((pixels >= self.lower_bound) & (pixels <= self.upper_bound), axis=-1)
        masked_pixels = pixels[bounds_mask]
        avg = masked_pixels.mean(axis=0) if masked_pixels.shape[0] > 0 else pixels.mean(axis=0)
        b_avg, g_avg, r_avg = avg
        if g_avg == 0:
            g_avg = 1e-6
        return {
            "single_illuminant": (r_avg / g_avg, b_avg / g_avg),
            "multi_illuminants": None,
            "illuminant_map": None,
            "estimated_srgb_image": None,
        }
