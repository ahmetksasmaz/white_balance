import numpy as np
import cv2 as cv
from .white_balance_algorithm import WhiteBalanceAlgorithm
from chromatic_adaptation import WhitePoint

class MaxRGB(WhiteBalanceAlgorithm):
    def __init__(self):
        pass

    def _apply_internal(self, image):
        illuminant = np.max(image, axis=(0, 1))
        gain = np.array([1.0, 1.0, 1.0]) / illuminant
        image = image * gain
        white_point = WhitePoint(illuminant, "SRGB")
        return image, {"white_point": white_point}