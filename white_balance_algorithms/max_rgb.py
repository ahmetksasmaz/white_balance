import numpy as np
import cv2 as cv
from .white_balance_algorithm import WhiteBalanceAlgorithm
from chromatic_adaptation import WhitePoint

class MaxRGB(WhiteBalanceAlgorithm):
    def __init__(self):
        pass

    def _estimate_global_illuminant_internal(self, image):
        illuminant = np.max(image, axis=(0, 1))
        white_point = WhitePoint(illuminant, "SRGB")
        return white_point