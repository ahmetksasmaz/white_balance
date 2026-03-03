import cv2 as cv
import numpy as np
from datasets.data import Data

class WhiteBalanceAlgorithm:
    def __init__(self):
        super().__init__()
    
    def estimate(self, data, process_masked=False):
        return self._estimate(data, process_masked)
    
    def _estimate(self, data, process_masked=False):
        raise NotImplementedError("This method should be implemented by subclasses")

    def _get_pixels(self, image, data, process_masked):
        """Helper: returns (N, 3) array of valid pixels based on mask.
        If process_masked=True and data has a mask, returns only unmasked pixels.
        Otherwise returns all pixels reshaped to (H*W, 3).
        """
        if process_masked and data.get_mask() is not None:
            mask = data.get_mask()  # (H, W) boolean, True=valid
            return image[mask]  # (N, 3)
        else:
            return image.reshape(-1, 3)  # (H*W, 3)