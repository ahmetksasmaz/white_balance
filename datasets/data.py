import cv2 as cv
import numpy as np
from error_metrics.single_illuminant_error_metrics import SingleIlluminantErrorMetrics
from error_metrics.multi_illuminant_error_metrics import MultiIlluminantErrorMetrics
from error_metrics.image_error_metrics import ImageErrorMetrics

class Data:
    def __init__(self):
        self.image_name = None
        self.raw_image = None
        self.srgb_image = None
        self.image_dimensions = (None, None, None)
        self.illuminants = None
        self.illuminant_map = None
        self.scene_data = None
        self.quantization = None
        self.exposure_values = {
            "exposure_time": None,
            "iso": None,
            "aperture": None
        }
        self.sensor_linear = True
        self.multi_illuminant = False
        self.mask = None  # Boolean array (H, W): True = valid pixel, False = masked (e.g. checkerboard)
        self.camera = None

    def set_image_name(self, image_name):
        self.image_name = image_name

    def set_camera(self, camera):
        self.camera = camera

    def get_camera(self):
        return self.camera

    def set_raw_image(self, raw_image):
        self.raw_image = raw_image
        self.image_dimensions = raw_image.shape
    
    def set_quantization(self, quantization):
        self.quantization = quantization

    def set_srgb_image(self, srgb_image):
        self.srgb_image = srgb_image
        self.sensor_linear = False
    
    def get_srgb_image(self):
        return self.srgb_image

    def set_illuminants(self, illuminants):
        self.illuminants = illuminants
    
    def set_illuminant_map(self, illuminant_map):
        self.illuminant_map = illuminant_map
        self.multi_illuminant = True
    
    def set_scene_data(self, scene_data):
        self.scene_data = scene_data
    
    def set_exposure_values(self, exposure_time, iso, aperture):
        self.exposure_values["exposure_time"] = exposure_time
        self.exposure_values["iso"] = iso
        self.exposure_values["aperture"] = aperture

    def set_mask(self, mask):
        """Set a boolean mask (H, W). True = valid pixel, False = masked pixel."""
        self.mask = mask

    def get_mask(self):
        return self.mask

    def get_data(self):
        return {
            "image_name": self.image_name,
            "camera": self.camera,
            "raw_image": self.get_raw_image(),
            "image_dimensions": self.image_dimensions,
            "illuminants": self.illuminants,
            "illuminant_map": self.illuminant_map,
            "scene_data": self.scene_data,
            "exposure_values": self.get_exposure_values(),
            "srgb_image": self.get_srgb_image(),
            "sensor_linear": self.sensor_linear,
            "multi_illuminant": self.multi_illuminant,
            "quantization": self.quantization,
            "mask": self.mask
        }
    
    def get_image_name(self):
        return self.image_name

    def get_raw_image(self):
        return self.raw_image
    
    def get_quantization(self):
        return self.quantization

    def get_image_dimensions(self):
        return self.image_dimensions
    
    def get_illuminants(self):
        return self.illuminants
    
    def get_illuminant_map(self):
        return self.illuminant_map
    
    def get_scene_data(self):
        return self.scene_data
    
    def get_exposure_values(self):
        return self.exposure_values

    def is_multi_illuminant(self):
        return self.multi_illuminant
    
    def _valid_illuminants(self):
        if self.illuminants is None:
            return []
        if isinstance(self.illuminants, dict):
            return [v for v in self.illuminants.values() if v is not None]
        if isinstance(self.illuminants, (list, tuple)):
            return [v for v in self.illuminants if v is not None]
        return [self.illuminants]

    def _build_flat_illuminant_map(self, single_illuminant, reference_map=None, target_shape=None):
        if single_illuminant is None:
            return None
        r_g, b_g = single_illuminant
        if reference_map is not None:
            if reference_map.ndim == 3 and reference_map.shape[2] == 2:
                flat_map = np.zeros_like(reference_map, dtype=np.float32)
                flat_map[..., 0] = r_g
                flat_map[..., 1] = b_g
                return flat_map
            if reference_map.ndim == 3 and reference_map.shape[2] == 3:
                r = np.full(reference_map.shape[:2], r_g, dtype=np.float32)
                g = np.full(reference_map.shape[:2], 1.0, dtype=np.float32)
                b = np.full(reference_map.shape[:2], b_g, dtype=np.float32)
                return np.stack([r, g, b], axis=-1)
            raise ValueError(f"Unsupported reference illuminant map shape for flat map: {reference_map.shape}")
        if target_shape is not None:
            if len(target_shape) == 3 and target_shape[2] == 2:
                flat_map = np.zeros(target_shape, dtype=np.float32)
                flat_map[..., 0] = r_g
                flat_map[..., 1] = b_g
                return flat_map
            if len(target_shape) == 3 and target_shape[2] == 3:
                r = np.full(target_shape[:2], r_g, dtype=np.float32)
                g = np.full(target_shape[:2], 1.0, dtype=np.float32)
                b = np.full(target_shape[:2], b_g, dtype=np.float32)
                return np.stack([r, g, b], axis=-1)
            raise ValueError(f"Unsupported target shape for flat map: {target_shape}")
        if self.raw_image is not None:
            shape = self.raw_image.shape[:2]
            flat_map = np.zeros((shape[0], shape[1], 2), dtype=np.float32)
            flat_map[..., 0] = r_g
            flat_map[..., 1] = b_g
            return flat_map
        return None

    def compute_error_metrics(self, estimations):
        errors_metrics = {
            "single_illuminant_errors": None,
            "multi_illuminant_errors": None,
            "image_errors": None
        }

        gt_illuminants = self._valid_illuminants()
        gt_map = self.illuminant_map
        est_single = estimations.get("single_illuminant")
        est_map = estimations.get("illuminant_map")

        # single-algorithm on single-illuminant data
        if est_single is not None and len(gt_illuminants) == 1 and not self.multi_illuminant:
            single_illuminant_metrics = SingleIlluminantErrorMetrics(gt_illuminants[0])
            errors_metrics["single_illuminant_errors"] = single_illuminant_metrics.errors(est_single)

        # single-algorithm on multi-illuminant data -> evaluate as flat estimated map
        if est_single is not None and self.multi_illuminant:
            if gt_map is None:
                print("Warning: Multi-illuminant ground truth map unavailable; cannot compute multi-illuminant errors from single illuminant estimation.")
            else:
                flat_est_map = self._build_flat_illuminant_map(est_single, reference_map=gt_map)
                multi_illuminant_metrics = MultiIlluminantErrorMetrics(gt_illuminants, gt_map)
                errors_metrics["multi_illuminant_errors"] = multi_illuminant_metrics.errors(None, flat_est_map)

        # multi-algorithm on multi-illuminant data
        if est_map is not None and self.multi_illuminant:
            if gt_map is None:
                print("Warning: Ground truth illuminant map unavailable; cannot compute multi-illuminant errors.")
            else:
                multi_illuminant_metrics = MultiIlluminantErrorMetrics(gt_illuminants, gt_map)
                errors_metrics["multi_illuminant_errors"] = multi_illuminant_metrics.errors(None, est_map)

        # multi-algorithm on single-illuminant data -> compare estimated map to flat GT map
        if est_map is not None and not self.multi_illuminant and len(gt_illuminants) == 1:
            flat_gt_map = self._build_flat_illuminant_map(gt_illuminants[0], reference_map=est_map)
            multi_illuminant_metrics = MultiIlluminantErrorMetrics(gt_illuminants, flat_gt_map)
            errors_metrics["multi_illuminant_errors"] = multi_illuminant_metrics.errors(None, est_map)

        if estimations.get("estimated_srgb_image") is not None:
            if self.sensor_linear:
                print("Warning: Estimated sRGB image provided for sensor-linear data. Cannot compute image error metrics due to lack of ground truth sRGB image.")
            else:
                image_metrics = ImageErrorMetrics(self.srgb_image)
                errors_metrics["image_errors"] = image_metrics.errors(estimations["estimated_srgb_image"])

        return errors_metrics