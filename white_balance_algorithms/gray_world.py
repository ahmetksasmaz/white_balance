import numpy as np
import cv2 as cv
from .white_balance_algorithm import WhiteBalanceAlgorithm
from chromatic_adaptation import WhitePoint

class GrayWorld(WhiteBalanceAlgorithm):
    def __init__(self):
        pass

    def _apply_internal(self, image):
        illuminant = np.mean(image, axis=(0, 1))
        gain = np.array([0.5, 0.5, 0.5]) / illuminant
        image = image * gain
        white_point = WhitePoint(illuminant, "SRGB")
        return image, {"white_point": white_point}