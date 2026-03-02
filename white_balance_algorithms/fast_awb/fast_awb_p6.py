import cv2 as cv
import numpy as np
from datasets.data import Data
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class FastAWBP6(WhiteBalanceAlgorithm):
    def __init__(self):
        super().__init__()
        self.p = 6
    
    def _estimate(self, data):
        image = data.get_raw_image()  # image is normalized to 0-1
        h, w = image.shape[:2]
        downsampled = cv.resize(image, (w // 4, h // 4), interpolation=cv.INTER_LINEAR)
        
        ycbcr = cv.cvtColor(downsampled, cv.COLOR_BGR2YCrCb).astype(np.float64)
        Y_max = np.max(ycbcr[:, :, 0])
        
        Cr = ycbcr[:, :, 1] - 0.5
        Cb = ycbcr[:, :, 2] - 0.5
        
        sigma = 1.0
        g_x = np.exp(-np.power(Cr + Cb, 2) / (2 * (sigma ** 2)))
        
        rho = g_x / np.sum(g_x)
        
        S_bar = np.zeros(3)
        for k in range(3):
            S_bar[k] = np.power(np.sum(rho * np.power(downsampled[:, :, k], self.p)), 1.0 / self.p)

        # S_bar is the estimated BGR illuminant color
        # Avoid division by zero
        if S_bar[1] == 0:
            S_bar[1] = 1e-6
        r_g = S_bar[2] / S_bar[1]
        b_g = S_bar[0] / S_bar[1]
        return {
            "single_illuminant": (r_g, b_g),
            "multi_illuminants": None,
            "illuminant_map": None,
            "estimated_srgb_image": None
        }