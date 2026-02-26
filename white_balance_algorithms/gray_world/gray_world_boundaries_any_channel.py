import cv2 as cv
import numpy as np
from datasets.data import Data
from white_balance_algorithms.single_illuminant_estimation import SingleIlluminantEstimationAlgorithm

class GrayWorldBoundariesAnyChannel(SingleIlluminantEstimationAlgorithm):
    def __init__(self, lower_bound=0.05, upper_bound=0.95):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def _estimate(self, data):
        image = data.get_raw_image()  # image is normalized to 0-1
        # Compute the average color between the lower and upper bounds (0-1)
        lower_bound_value = self.lower_bound
        upper_bound_value = self.upper_bound
        # Exclude pixels where all channels are below lower bound or all channels are above upper bound
        all_below = np.all(image < lower_bound_value, axis=-1)
        all_above = np.all(image > upper_bound_value, axis=-1)
        mask = ~(all_below | all_above)
        # Use mask to select valid pixels
        masked_pixels = image[mask]
        num_masked = masked_pixels.shape[0]
        if num_masked == 0:
            # Fallback: use mean over all pixels
            avg_color_per_channel = np.mean(image, axis=(0, 1))
        else:
            avg_color_per_channel = masked_pixels.mean(axis=0)
        b_avg, g_avg, r_avg = avg_color_per_channel
        # Avoid division by zero
        if g_avg == 0:
            g_avg = 1e-6
        r_g = r_avg / g_avg
        b_g = b_avg / g_avg
        return (r_g, b_g)