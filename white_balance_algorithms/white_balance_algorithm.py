import cv2 as cv    
import numpy as np
from chromatic_adaptation import D65_WHITE_POINT, WhitePoint

class WhiteBalanceAlgorithm:
    def __init__(self):
        pass

    def estimate_global_illuminant(self, image: np.ndarray) -> WhitePoint:
        return self._estimate_global_illuminant_internal(image)

    def estimate_multiple_illuminants(self, image: np.ndarray) -> list[WhitePoint]:
        return self._estimate_multiple_illuminants_internal(image)

    def estimate_illuminant_map(self, image: np.ndarray) -> list[list[WhitePoint]]:
        return self._estimate_illuminant_map_internal(image)

    def _estimate_global_illuminant_internal(self, image: np.ndarray) -> WhitePoint:
        return D65_WHITE_POINT

    def _estimate_multiple_illuminants_internal(self, image: np.ndarray) -> list[WhitePoint]:
        return [D65_WHITE_POINT]

    def _estimate_illuminant_map_internal(self, image: np.ndarray) -> list[list[WhitePoint]]:
        return np.full(image.shape[:2], D65_WHITE_POINT)