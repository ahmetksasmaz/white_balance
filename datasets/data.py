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

    def set_image_name(self, image_name):
        self.image_name = image_name

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
    
    def compute_error_metrics(self, estimations):
        errors_metrics = {
            "single_illuminant_errors": None,
            "multi_illuminant_errors": None,
            "image_errors": None
        }
        if estimations["single_illuminant"] is not None:
            if self.multi_illuminant == True:
                print("Warning: Single illuminant estimation provided for multi-illuminant data. Taking first illuminant for error computation.")
            if self.illuminants is None:
                print("Warning: Ground truth illuminants not available for error computation.")
            else:
                first_gt_illuminant = None
                for key in self.illuminants.keys():
                    if self.illuminants[key] is not None:
                        first_gt_illuminant = self.illuminants[key]
                        break
                single_illuminant_metrics = SingleIlluminantErrorMetrics(first_gt_illuminant)
                errors_metrics["single_illuminant_errors"] = single_illuminant_metrics.errors(estimations["single_illuminant"])
        if estimations["multi_illuminants"] is not None:
            if self.multi_illuminant == False:
                print("Warning: Multi-illuminant estimation provided for single-illuminant data. Cannot compute multi-illuminant error metrics due to lack of illuminant map.")
            else:
                multi_illuminant_metrics = MultiIlluminantErrorMetrics(self.illuminant_map)
                errors_metrics["multi_illuminant_errors"] = multi_illuminant_metrics.errors(estimations["multi_illuminants"])
        if estimations["estimated_srgb_image"] is not None:
            if self.sensor_linear == True:
                print("Warning: Estimated sRGB image provided for sensor-linear data. Cannot compute image error metrics due to lack of ground truth sRGB image.")
            else:
                image_metrics = ImageErrorMetrics(self.srgb_image)
                errors_metrics["image_errors"] = image_metrics.errors(estimations["estimated_srgb_image"])
        return errors_metrics