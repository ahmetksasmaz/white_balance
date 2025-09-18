import cv2 as cv    
import numpy as np
from chromatic_adaptation import D65_WHITE_POINT, WhitePoint

class WhiteBalanceAlgorithm:
    def __init__(self):
        pass

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        return self._apply_internal(image)
    
    def _apply_internal(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        return image, {}