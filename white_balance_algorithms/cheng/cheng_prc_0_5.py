import numpy as np
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class ChengPrc05(WhiteBalanceAlgorithm):
    def __init__(self):
        self.prc = 0.005

    def _estimate(self, data, process_masked=False):
        pixels = self._get_pixels(data.get_raw_image(), data, process_masked)
        pixels = pixels[np.all(pixels < 0.98, axis=1)]
        if pixels.shape[0] == 0:
            return {"single_illuminant": (1.0, 1.0), "multi_illuminants": None, "illuminant_map": None, "estimated_srgb_image": None}

        l = np.mean(pixels, axis=0)
        norm_l = np.linalg.norm(l)
        l = l / norm_l if norm_l > 0 else np.array([1, 1, 1]) / np.sqrt(3)

        data_p = pixels @ l
        idx = np.argsort(data_p)
        n = pixels.shape[0]
        n_sel = int(np.ceil(n * self.prc))
        n_bottom = int(np.floor(n * (1.0 - self.prc)))
        data_selected = np.vstack((pixels[idx[:n_sel]], pixels[idx[n_bottom-1:]]))

        if data_selected.shape[0] == 0:
            b_est, g_est, r_est = l
        else:
            _, eigenvectors = np.linalg.eigh(data_selected.T @ data_selected)
            b_est, g_est, r_est = np.abs(eigenvectors[:, -1])

        if g_est == 0:
            g_est = 1e-6
        return {
            "single_illuminant": (r_est / g_est, b_est / g_est),
            "multi_illuminants": None,
            "illuminant_map": None,
            "estimated_srgb_image": None,
        }
