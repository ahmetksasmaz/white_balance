import cv2 as cv
import numpy as np

class ImageErrorMetrics:
    def __init__(self, ground_truth_srgb_image):
        super().__init__()
        self.ground_truth_srgb_image = ground_truth_srgb_image
    
    def errors(self, estimated_srgb_image):
        # Compute error metrics between ground truth and estimated sRGB images
        mse = self._mse(estimated_srgb_image)
        delta_e = self._ciede_2000(estimated_srgb_image)
        return {
            "mse": mse,
            "delta_e": delta_e
        }
        raise NotImplementedError("Multi-illuminant error metrics are not implemented yet")
    
    def _mse(self, estimated_srgb_image):
        return np.mean((self.ground_truth_srgb_image.astype(np.float32) - estimated_srgb_image.astype(np.float32)) ** 2)
    
    def _ciede_2000(self, estimated_srgb_image):
        # Convert images to Lab color space and decode OpenCV encoding
        gt_lab_cv  = cv.cvtColor(self.ground_truth_srgb_image, cv.COLOR_BGR2LAB).astype(np.float32)
        est_lab_cv = cv.cvtColor(estimated_srgb_image,         cv.COLOR_BGR2LAB).astype(np.float32)

        # OpenCV encodes L in [0,255] (from [0,100]) and a,b shifted by +128
        L1 = gt_lab_cv[:, :, 0] * 100.0 / 255.0;  a1 = gt_lab_cv[:, :, 1] - 128.0;  b1 = gt_lab_cv[:, :, 2] - 128.0
        L2 = est_lab_cv[:, :, 0] * 100.0 / 255.0; a2 = est_lab_cv[:, :, 1] - 128.0; b2 = est_lab_cv[:, :, 2] - 128.0

        # C*ab and G factor
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_mean7 = ((C1 + C2) / 2.0) ** 7
        G = 0.5 * (1.0 - np.sqrt(C_mean7 / (C_mean7 + 25.0**7)))

        # Adjusted a', C', h'
        a1p = (1.0 + G) * a1
        a2p = (1.0 + G) * a2
        C1p = np.sqrt(a1p**2 + b1**2)
        C2p = np.sqrt(a2p**2 + b2**2)
        h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
        h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

        # Delta L', Delta C'
        dLp = L2 - L1
        dCp = C2p - C1p

        # Delta h' (vectorized, zero when either C' == 0)
        dhp = h2p - h1p
        dhp = np.where(dhp >  180.0, dhp - 360.0, dhp)
        dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
        dhp = np.where(C1p * C2p == 0.0, 0.0, dhp)
        dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

        # Arithmetic means of L', C', h'
        Lp_mean = (L1 + L2) / 2.0
        Cp_mean = (C1p + C2p) / 2.0
        hp_mean = (h1p + h2p) / 2.0
        hp_mean = np.where(np.abs(h1p - h2p) > 180.0, hp_mean + 180.0, hp_mean)
        hp_mean = hp_mean % 360.0
        hp_mean = np.where(C1p * C2p == 0.0, h1p + h2p, hp_mean)

        # T weighting function
        T = (1.0
             - 0.17 * np.cos(np.radians(hp_mean - 30.0))
             + 0.24 * np.cos(np.radians(2.0 * hp_mean))
             + 0.32 * np.cos(np.radians(3.0 * hp_mean + 6.0))
             - 0.20 * np.cos(np.radians(4.0 * hp_mean - 63.0)))

        # Weighting functions
        SL = 1.0 + (0.015 * (Lp_mean - 50.0)**2) / np.sqrt(20.0 + (Lp_mean - 50.0)**2)
        SC = 1.0 + 0.045 * Cp_mean
        SH = 1.0 + 0.015 * Cp_mean * T

        # Rotation term
        Cp_mean7 = Cp_mean ** 7
        RC = 2.0 * np.sqrt(Cp_mean7 / (Cp_mean7 + 25.0**7))
        delta_ro = 30.0 * np.exp(-((hp_mean - 275.0) / 25.0)**2)
        RT = -np.sin(np.radians(2.0 * delta_ro)) * RC

        # Final Delta E 2000
        dE = np.sqrt(
            (dLp / SL)**2 +
            (dCp / SC)**2 +
            (dHp / SH)**2 +
            RT * (dCp / SC) * (dHp / SH)
        )
        return float(np.mean(dE))
    