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
        self.exposure_values = {"exposure_time": None, "iso": None, "aperture": None}
        self.sensor_linear = True
        self.multi_illuminant = False
        self.mask = None
        self.camera = None

    def set_image_name(self, image_name): self.image_name = image_name
    def set_camera(self, camera): self.camera = camera
    def get_camera(self): return self.camera
    def set_raw_image(self, raw_image):
        self.raw_image = raw_image
        self.image_dimensions = raw_image.shape
    def set_quantization(self, quantization): self.quantization = quantization
    def set_srgb_image(self, srgb_image):
        self.srgb_image = srgb_image
        self.sensor_linear = False
    def get_srgb_image(self): return self.srgb_image
    def set_illuminants(self, illuminants): self.illuminants = illuminants
    def set_illuminant_map(self, illuminant_map):
        self.illuminant_map = illuminant_map
        self.multi_illuminant = True
    def set_scene_data(self, scene_data): self.scene_data = scene_data
    def set_exposure_values(self, exposure_time, iso, aperture):
        self.exposure_values["exposure_time"] = exposure_time
        self.exposure_values["iso"] = iso
        self.exposure_values["aperture"] = aperture
    def set_mask(self, mask): self.mask = mask
    def get_mask(self): return self.mask
    def resize(self, resize_factor):
        if resize_factor is None or resize_factor <= 1 or self.raw_image is None:
            return

        target_width = max(1, self.raw_image.shape[1] // resize_factor)
        target_height = max(1, self.raw_image.shape[0] // resize_factor)
        target_size = (target_width, target_height)

        self.raw_image = cv.resize(self.raw_image, target_size, interpolation=cv.INTER_AREA)

        if self.mask is not None:
            self.mask = cv.resize(self.mask.astype(np.uint8), target_size, interpolation=cv.INTER_NEAREST).astype(bool)

        if self.illuminant_map is not None:
            self.illuminant_map = cv.resize(self.illuminant_map.astype(np.float32), target_size, interpolation=cv.INTER_LINEAR)

        if self.srgb_image is not None:
            self.srgb_image = cv.resize(self.srgb_image, target_size, interpolation=cv.INTER_AREA)

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
            "mask": self.mask,
        }
    def get_image_name(self): return self.image_name
    def get_raw_image(self): return self.raw_image
    def get_quantization(self): return self.quantization
    def get_image_dimensions(self): return self.image_dimensions
    def get_illuminants(self): return self.illuminants
    def get_illuminant_map(self): return self.illuminant_map
    def get_scene_data(self): return self.scene_data
    def get_exposure_values(self): return self.exposure_values
    def is_multi_illuminant(self): return self.multi_illuminant

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
        shape = reference_map.shape if reference_map is not None else target_shape
        if shape is None and self.raw_image is not None:
            shape = (self.raw_image.shape[0], self.raw_image.shape[1], 2)
        if shape is None:
            return None
        if len(shape) == 3 and shape[2] == 2:
            flat_map = np.zeros(shape, dtype=np.float32)
            flat_map[..., 0] = r_g
            flat_map[..., 1] = b_g
            return flat_map
        if len(shape) == 3 and shape[2] == 3:
            return np.stack([
                np.full(shape[:2], r_g, dtype=np.float32),
                np.full(shape[:2], 1.0, dtype=np.float32),
                np.full(shape[:2], b_g, dtype=np.float32),
            ], axis=-1)
        raise ValueError(f"Unsupported shape for flat illuminant map: {shape}")

    def compute_error_metrics(self, estimations):
        errors_metrics = {"single_illuminant_errors": None, "multi_illuminant_errors": None, "image_errors": None}
        gt_illuminants = self._valid_illuminants()
        gt_map = self.illuminant_map
        est_single = estimations.get("single_illuminant")
        est_map = estimations.get("illuminant_map")

        if est_map is not None:
            # Algorithm produces a map: always compare map vs map using mean angular error.
            # If GT has only a single illuminant, build a flat map so the comparison is still pixel-wise.
            if gt_map is not None:
                effective_gt_map = gt_map
            elif len(gt_illuminants) == 1:
                effective_gt_map = self._build_flat_illuminant_map(gt_illuminants[0], reference_map=est_map)
            else:
                effective_gt_map = None
            if effective_gt_map is not None:
                errors_metrics["multi_illuminant_errors"] = MultiIlluminantErrorMetrics(gt_illuminants, effective_gt_map).errors(None, est_map)
            else:
                print("Warning: No ground truth available to compare against estimated illuminant map.")
        elif est_single is not None:
            if not self.multi_illuminant and len(gt_illuminants) == 1:
                errors_metrics["single_illuminant_errors"] = SingleIlluminantErrorMetrics(gt_illuminants[0]).errors(est_single)
            elif self.multi_illuminant and gt_map is not None:
                flat_est_map = self._build_flat_illuminant_map(est_single, reference_map=gt_map)
                errors_metrics["multi_illuminant_errors"] = MultiIlluminantErrorMetrics(gt_illuminants, gt_map).errors(None, flat_est_map)
            else:
                print("Warning: Multi-illuminant ground truth map unavailable; cannot compare single-illuminant estimation against multi-illuminant GT.")

        if estimations.get("estimated_srgb_image") is not None:
            if self.sensor_linear:
                print("Warning: Estimated sRGB image provided for sensor-linear data. Cannot compute image error metrics due to lack of ground truth sRGB image.")
            else:
                errors_metrics["image_errors"] = ImageErrorMetrics(self.srgb_image).errors(estimations["estimated_srgb_image"])

        return errors_metrics
