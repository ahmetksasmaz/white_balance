import cv2 as cv
import numpy as np

class SingleIlluminantErrorMetrics:
    def __init__(self, ground_truth_illuminant):
        super().__init__()
        self.ground_truth_illuminant = ground_truth_illuminant
        self.ground_truth_vector = self._chromaticity_to_rgb(ground_truth_illuminant)
    
    def errors(self, estimated_illuminant):
        angular = self._angular_error(estimated_illuminant, self.ground_truth_illuminant)
        return {
            "angular_error": angular
        }

    def _chromaticity_to_rgb(self, chromaticity):
        r_g, b_g = chromaticity
        g = 1.0
        r = r_g * g
        b = b_g * g
        # Chromaticity normalization: r + g + b = 1
        total = r + g + b
        if total == 0:
            # Avoid division by zero, return neutral gray
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)
        r_norm = r / total
        g_norm = g / total
        b_norm = b / total
        return np.array([r_norm, g_norm, b_norm], dtype=np.float32)

    def _angular_error(self, estimated_illuminant, ground_truth_illuminant):
        estimated_vector = self._chromaticity_to_rgb(estimated_illuminant)
        dot_product = np.dot(estimated_vector, self.ground_truth_vector)
        norm_estimated = np.linalg.norm(estimated_vector)
        norm_ground_truth = np.linalg.norm(self.ground_truth_vector)
        if norm_estimated == 0 or norm_ground_truth == 0:
            return 0.0
        cos_theta = dot_product / (norm_estimated * norm_ground_truth)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        return angle_deg