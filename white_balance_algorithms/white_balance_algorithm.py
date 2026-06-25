import numpy as np
from datasets.data import Data

class WhiteBalanceAlgorithm:
    def estimate(self, data, process_masked=False):
        estimations = self._estimate(data, process_masked)
        raw_image = data.get_raw_image()

        estimations.setdefault("single_illuminant", None)
        estimations.setdefault("multi_illuminants", None)
        estimations.setdefault("illuminant_map", None)
        estimations.setdefault("estimated_srgb_image", None)
        estimations.setdefault("single_illuminant_corrected_raw_image", None)
        estimations.setdefault("multi_illuminant_corrected_raw_image", None)

        if raw_image is not None:
            if estimations["single_illuminant"] is not None:
                estimations["single_illuminant_corrected_raw_image"] = self._apply_von_kries_single(raw_image, estimations["single_illuminant"])
            if estimations["illuminant_map"] is not None:
                estimations["multi_illuminant_corrected_raw_image"] = self._apply_von_kries_map(raw_image, estimations["illuminant_map"])

        return estimations

    def _estimate(self, data, process_masked=False):
        raise NotImplementedError

    def _get_pixels(self, image, data, process_masked):
        if process_masked and data.get_mask() is not None:
            return image[data.get_mask()]
        return image.reshape(-1, 3)

    def _apply_von_kries_single(self, raw_image, single_illuminant):
        try:
            r_g, b_g = single_illuminant
            r_scale = 1.0 / float(r_g) if float(r_g) != 0 else 0.0
            b_scale = 1.0 / float(b_g) if float(b_g) != 0 else 0.0
            scale = np.array([b_scale, 1.0, r_scale], dtype=np.float32)
            return np.clip(raw_image.astype(np.float32) * scale.reshape((1, 1, 3)), 0.0, 1.0)
        except Exception:
            return None

    def _apply_von_kries_map(self, raw_image, illuminant_map):
        if illuminant_map is None or raw_image is None:
            return None
        if illuminant_map.ndim == 3:
            if illuminant_map.shape[2] == 3:
                illuminant = illuminant_map.astype(np.float32)
                with np.errstate(divide='ignore', invalid='ignore'):
                    inv_illuminant = np.where(illuminant != 0, 1.0 / illuminant, 0.0)
                return np.clip(raw_image.astype(np.float32) * inv_illuminant, 0.0, 1.0)
            elif illuminant_map.shape[2] == 2:
                r_g = illuminant_map[..., 0].astype(np.float32)
                b_g = illuminant_map[..., 1].astype(np.float32)
                with np.errstate(divide='ignore', invalid='ignore'):
                    inv_r = np.where(r_g != 0, 1.0 / r_g, 0.0)
                    inv_b = np.where(b_g != 0, 1.0 / b_g, 0.0)
                inv_map = np.stack([inv_b, np.ones_like(inv_b), inv_r], axis=-1)
                return np.clip(raw_image.astype(np.float32) * inv_map, 0.0, 1.0)
        return None
