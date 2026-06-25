import numpy as np

class MultiIlluminantErrorMetrics:
    def __init__(self, ground_truth_illuminants, ground_truth_illuminant_map):
        self.ground_truth_illuminants = ground_truth_illuminants
        self.ground_truth_illuminant_map = ground_truth_illuminant_map

    def errors(self, estimated_illuminants, estimated_illuminant_map):
        return {"mean_angular_error": float(np.mean(self._angular_error_map(estimated_illuminant_map, self.ground_truth_illuminant_map)))}

    def _chromaticity_to_rgb_map(self, chromaticity_map):
        chromaticity_map = np.asarray(chromaticity_map, dtype=np.float32)

        if chromaticity_map.ndim == 3 and chromaticity_map.shape[2] == 3:
            norm = np.linalg.norm(chromaticity_map, axis=-1, keepdims=True)
            return chromaticity_map / np.where(norm == 0, 1.0, norm)

        if chromaticity_map.shape[-1] == 2:
            if chromaticity_map.ndim == 1:
                r_g, b_g = chromaticity_map
                total = r_g + 1.0 + b_g
                return np.array([r_g / total, 1.0 / total, b_g / total], dtype=np.float32) if total != 0 else np.array([1/3, 1/3, 1/3], dtype=np.float32)
            r_g = chromaticity_map[..., 0]
            b_g = chromaticity_map[..., 1]
            total = r_g + 1.0 + b_g
            total = np.where(total == 0, 1.0, total)
            return np.stack([r_g / total, np.ones_like(r_g) / total, b_g / total], axis=-1)

        raise ValueError(f"Unsupported chromaticity map shape {chromaticity_map.shape}.")

    def _angular_error_map(self, estimated_illuminant_map, ground_truth_illuminant_map):
        est = self._chromaticity_to_rgb_map(estimated_illuminant_map)
        gt = self._chromaticity_to_rgb_map(ground_truth_illuminant_map)
        if est.shape != gt.shape:
            raise ValueError(f"Shape mismatch: {est.shape} vs {gt.shape}.")
        dot = np.sum(est * gt, axis=-1)
        denom = np.linalg.norm(est, axis=-1) * np.linalg.norm(gt, axis=-1)
        cos_theta = np.zeros_like(dot, dtype=np.float32)
        valid = denom > 0
        cos_theta[valid] = dot[valid] / denom[valid]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_theta))
        angle_deg[~valid] = 0.0
        return angle_deg
