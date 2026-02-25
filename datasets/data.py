import cv2 as cv
import numpy as np

class Data:
    def __init__(self):
        self.image_name = None
        self.raw_image = None
        self.image_dimensions = (None, None, None)
        self.illuminants = None
        self.illuminant_map = None
        self.scene_data = None
        self.exposure_values = {
            "exposure_time": None,
            "iso": None,
            "aperture": None
        }
        self.multi_illuminant = False

    def set_image_name(self, image_name):
        self.image_name = image_name

    def set_raw_image(self, raw_image):
        self.raw_image = raw_image
        self.image_dimensions = raw_image.shape
    
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

    def get_data(self):
        return {
            "image_name": self.image_name,
            "raw_image": self.get_raw_image(),
            "image_dimensions": self.image_dimensions,
            "illuminants": self.illuminants,
            "illuminant_map": self.illuminant_map,
            "scene_data": self.scene_data,
            "exposure_values": self.get_exposure_values()
        }
    
    def get_image_name(self):
        return self.image_name

    def get_raw_image(self):
        return self.raw_image
    
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