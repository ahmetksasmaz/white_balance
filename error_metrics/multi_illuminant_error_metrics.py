import cv2 as cv
import numpy as np

class MultiIlluminantErrorMetrics:
    def __init__(self, ground_truth_illuminants, ground_truth_illuminant_map):
        super().__init__()
        self.ground_truth_illuminants = ground_truth_illuminants
        self.ground_truth_illuminant_map = ground_truth_illuminant_map
    
    def errors(self, estimated_illuminants, estimated_illuminant_map):
        raise NotImplementedError("Multi-illuminant error metrics are not implemented yet")