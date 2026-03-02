import cv2 as cv
import numpy as np
from datasets.data import Data

class WhiteBalanceAlgorithm:
    def __init__(self):
        super().__init__()
    
    def estimate(self, data):
        return self._estimate(data)
    
    def _estimate(self, data):
        raise NotImplementedError("This method should be implemented by subclasses")