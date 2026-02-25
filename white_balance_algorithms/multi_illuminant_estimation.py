import cv2 as cv
import numpy as np
from datasets.data import Data

class MultiIlluminantEstimationAlgorithm:
    def __init__(self):
        super().__init__()
    
    def estimate(self, data):
        if not data.is_multi_illuminant():
            raise ValueError("Input data does not contain multiple illuminants, but this algorithm is designed for multi-illuminant estimation.")
        return self._estimate(data)
    
    def _estimate(self, data):
        raise NotImplementedError("This method should be implemented by subclasses")