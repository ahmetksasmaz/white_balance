import numpy as np

class SingleIlluminantErrorMetrics:
    def __init__(self, ground_truth_illuminant):
        self.ground_truth_illuminant = ground_truth_illuminant
        self.ground_truth_vector = self._chromaticity_to_rgb(ground_truth_illuminant)

    def errors(self, estimated_illuminant):
        return {"angular_error": self._angular_error(estimated_illuminant)}

    def _chromaticity_to_rgb(self, chromaticity):
        r_g, b_g = chromaticity
        total = r_g + 1.0 + b_g
        if total == 0:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)
        return np.array([r_g / total, 1.0 / total, b_g / total], dtype=np.float32)

    def _angular_error(self, estimated_illuminant):
        est = self._chromaticity_to_rgb(estimated_illuminant)
        norm_est = np.linalg.norm(est)
        norm_gt = np.linalg.norm(self.ground_truth_vector)
        if norm_est == 0 or norm_gt == 0:
            return 0.0
        cos_theta = np.clip(np.dot(est, self.ground_truth_vector) / (norm_est * norm_gt), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))
