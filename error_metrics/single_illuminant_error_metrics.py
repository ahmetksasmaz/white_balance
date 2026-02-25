import cv2 as cv
import numpy as np

class SingleIlluminantErrorMetrics:
    def __init__(self, ground_truth_illuminant):
        super().__init__()
        self.ground_truth_illuminant = ground_truth_illuminant
        self.ground_truth_vector = self._chromaticity_to_rgb(ground_truth_illuminant)
    
    def errors(self, estimated_illuminant):
        angular = self._angular_error(estimated_illuminant, self.ground_truth_illuminant)
        # square = self._square_error(estimated_illuminant, self.ground_truth_illuminant)
        # absolute = self._absolute_error(estimated_illuminant, self.ground_truth_illuminant)
        # ciede2000 = self._ciede2000_error(estimated_illuminant, self.ground_truth_illuminant)
        return {
            "angular_error": angular,
            "square_error": "To Be Fixed", #square,
            "absolute_error": "To Be Fixed", #absolute,
            "ciede2000_error": "To Be Fixed", #ciede2000
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
    

    def _square_error(self, estimated_illuminant, ground_truth_illuminant):
        estimated_vector = self._chromaticity_to_rgb(estimated_illuminant) * 255.0
        ground_truth_vector = self.ground_truth_vector * 255.0
        error = np.sum((estimated_vector - ground_truth_vector) ** 2)
        return error

    def _absolute_error(self, estimated_illuminant, ground_truth_illuminant):
        estimated_vector = self._chromaticity_to_rgb(estimated_illuminant) * 255.0
        ground_truth_vector = self.ground_truth_vector * 255.0
        error = np.sum(np.abs(estimated_vector - ground_truth_vector))
        return error
    
    def _ciede2000_error(self, estimated_illuminant, ground_truth_illuminant):
        # Convert chromaticities to RGB
        estimated_rgb = self._chromaticity_to_rgb(estimated_illuminant)
        ground_truth_rgb = self._chromaticity_to_rgb(ground_truth_illuminant)
        # Reshape for cv2 (single pixel)
        estimated_rgb_reshaped = np.array([[estimated_rgb]], dtype=np.float32)
        ground_truth_rgb_reshaped = np.array([[ground_truth_rgb]], dtype=np.float32)

        # Convert RGB to Lab using OpenCV (assume RGB in [0, 1], scale to [0, 255])
        estimated_rgb_scaled = np.clip(estimated_rgb_reshaped * 255, 0, 255).astype(np.uint8)
        ground_truth_rgb_scaled = np.clip(ground_truth_rgb_reshaped * 255, 0, 255).astype(np.uint8)
        estimated_lab = cv.cvtColor(estimated_rgb_scaled, cv.COLOR_RGB2LAB)[0,0]
        ground_truth_lab = cv.cvtColor(ground_truth_rgb_scaled, cv.COLOR_RGB2LAB)[0,0]

        # CIEDE2000 calculation
        def ciede2000(lab1, lab2):
            # Implementation based on the formula from Sharma et al. (2005)
            L1, a1, b1 = lab1
            L2, a2, b2 = lab2
            avg_L = (L1 + L2) / 2.0
            C1 = np.sqrt(a1 ** 2 + b1 ** 2)
            C2 = np.sqrt(a2 ** 2 + b2 ** 2)
            avg_C = (C1 + C2) / 2.0
            G = 0.5 * (1 - np.sqrt(avg_C ** 7 / (avg_C ** 7 + 25 ** 7)))
            a1p = (1 + G) * a1
            a2p = (1 + G) * a2
            C1p = np.sqrt(a1p ** 2 + b1 ** 2)
            C2p = np.sqrt(a2p ** 2 + b2 ** 2)
            avg_Cp = (C1p + C2p) / 2.0
            h1p = np.degrees(np.arctan2(b1, a1p)) % 360
            h2p = np.degrees(np.arctan2(b2, a2p)) % 360
            deltahp = h2p - h1p
            if deltahp > 180:
                deltahp -= 360
            elif deltahp < -180:
                deltahp += 360
            deltaLp = L2 - L1
            deltaCp = C2p - C1p
            deltaHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(deltahp) / 2)
            avg_Lp = (L1 + L2) / 2.0
            avg_hp = (h1p + h2p) / 2.0
            if abs(h1p - h2p) > 180:
                avg_hp += 180
            avg_hp %= 360
            T = 1 - 0.17 * np.cos(np.radians(avg_hp - 30)) + 0.24 * np.cos(np.radians(2 * avg_hp)) + 0.32 * np.cos(np.radians(3 * avg_hp + 6)) - 0.20 * np.cos(np.radians(4 * avg_hp - 63))
            delta_ro = 30 * np.exp(-((avg_hp - 275) / 25) ** 2)
            Rc = 2 * np.sqrt(avg_Cp ** 7 / (avg_Cp ** 7 + 25 ** 7))
            Sl = 1 + (0.015 * (avg_Lp - 50) ** 2) / np.sqrt(20 + (avg_Lp - 50) ** 2)
            Sc = 1 + 0.045 * avg_Cp
            Sh = 1 + 0.015 * avg_Cp * T
            Rt = -np.sin(np.radians(2 * delta_ro)) * Rc
            deltaE = np.sqrt(
                (deltaLp / Sl) ** 2 +
                (deltaCp / Sc) ** 2 +
                (deltaHp / Sh) ** 2 +
                Rt * (deltaCp / Sc) * (deltaHp / Sh)
            )
            return deltaE

        return float(ciede2000(estimated_lab, ground_truth_lab))