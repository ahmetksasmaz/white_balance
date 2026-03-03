import cv2 as cv
import numpy as np
from datasets.data import Data
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

class FastAWBDefault(WhiteBalanceAlgorithm):
    def __init__(self):
        super().__init__()
        self.p = 1
    
    def _estimate(self, data, process_masked=False):
        image = data.get_raw_image()  # image is normalized to 0-1

        # Apply mask before downsampling: zero out masked pixels
        if process_masked and data.get_mask() is not None:
            mask = data.get_mask()
            image = image.copy()
            image[~mask] = 0

        h, w = image.shape[:2]
        downsampled = cv.resize(image, (w // 4, h // 4), interpolation=cv.INTER_LINEAR)
        
        # Downsample the mask to match
        if process_masked and data.get_mask() is not None:
            mask_ds = cv.resize(mask.astype(np.uint8), (w // 4, h // 4), interpolation=cv.INTER_NEAREST).astype(bool)
        else:
            mask_ds = np.ones((h // 4, w // 4), dtype=bool)

        ycbcr = cv.cvtColor(downsampled, cv.COLOR_BGR2YCrCb).astype(np.float64)
        
        Cr = ycbcr[:, :, 1] - 0.5
        Cb = ycbcr[:, :, 2] - 0.5
        
        sigma = 1.0
        g_x = np.exp(-np.power(Cr + Cb, 2) / (2 * (sigma ** 2)))
        
        # Zero out weights for masked pixels
        g_x[~mask_ds] = 0
        
        g_sum = np.sum(g_x)
        if g_sum == 0:
            g_sum = 1e-6
        rho = g_x / g_sum
        
        S_bar = np.zeros(3)
        for k in range(3):
            S_bar[k] = np.sum(rho * downsampled[:, :, k])

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