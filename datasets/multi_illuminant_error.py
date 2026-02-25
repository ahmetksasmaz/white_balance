import cv2 as cv
import numpy as np
from error import ErrorMetrics

class MultiIlluminantErrorMetrics(ErrorMetrics):
    def __init__(self, ground_truth_illuminants, ground_truth_illuminant_map):
        super().__init__()
        self.ground_truth_illuminants = ground_truth_illuminants
        self.ground_truth_illuminant_map = ground_truth_illuminant_map