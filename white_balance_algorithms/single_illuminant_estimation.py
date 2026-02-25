import cv2 as cv
import numpy as np
from datasets.data import Data

class SingleIlluminantEstimationAlgorithm:
    def __init__(self):
        super().__init__()
    
    def estimate(self, data):
        if data.is_multi_illuminant():
            raise ValueError("Input data contains multiple illuminants, but this algorithm is designed for single illuminant estimation.")
        return self._estimate(data)
    
    def _estimate(self, data):
        raise NotImplementedError("This method should be implemented by subclasses")