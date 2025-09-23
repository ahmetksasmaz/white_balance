import cv2 as cv
import numpy as np
from .configuration import *
from chromatic_adaptation import *

class LSMIData:
    def __init__(self, image_path : str, coeff_path : str, lights : list[list[float]], load_resized=False):
        self.image_path = image_path
        self.coeff_path = coeff_path
        self.lights = lights
        self.image = None
        self.gt_image = None
        self.coeff_map = None
        self.load_resized = load_resized

        image_path_splitted = self.image_path.split("/")
        self.place = image_path_splitted[-2]
        self.camera_model = image_path_splitted[-3]

        self.__parse_metadata()
        self.__load_image()
        self.__load_coeff()
        self.__construct_gt_image()
    
    def __parse_metadata(self):
        pass

    def __load_image(self):
        # Load the image using OpenCV
        self.image = cv.imread(self.image_path, cv.IMREAD_UNCHANGED)
        self.image = self.image.astype(np.float32)

        BLACK_LEVEL = None
        SATURATION_LEVEL = None
        if self.camera_model == "galaxy":
            BLACK_LEVEL = GALAXY_BLACK_LEVEL
            SATURATION_LEVEL = GALAXY_SATURATION_LEVEL
        elif self.camera_model == "nikon":
            BLACK_LEVEL = NIKON_BLACK_LEVEL
            SATURATION_LEVEL = NIKON_SATURATION_LEVEL
        elif self.camera_model == "sony":
            BLACK_LEVEL = SONY_BLACK_LEVEL
            SATURATION_LEVEL = SONY_SATURATION_LEVEL

        self.image = np.clip((self.image - BLACK_LEVEL) / (SATURATION_LEVEL - BLACK_LEVEL), 0, 1)

        if self.load_resized:
            orig_h, orig_w = self.image.shape[:2]
            new_w = int(orig_w * (512 / orig_h))
            self.image = cv.resize(self.image, (new_w, 512))
    
    def __load_coeff(self):
        self.coeff_map = np.load(self.coeff_path)
        self.coeff_map = self.coeff_map.astype(np.float32)
        if self.load_resized:
            orig_h, orig_w = self.coeff_map.shape[:2]
            new_w = int(orig_w * (512 / orig_h))
            self.coeff_map = cv.resize(self.coeff_map, (new_w, 512))

    def __construct_gt_image(self):
        illuminant__map = np.ones_like(self.image, dtype=np.float32)
        if len(self.lights) == 2:
            coeff1, coeff2 = cv.split(self.coeff_map)
            light_1 = np.array(self.lights[0], dtype=np.float32)
            light_2 = np.array(self.lights[1], dtype=np.float32)
            light_1_map = cv.merge([light_1[0]*coeff1, light_1[1]*coeff1, light_1[2]*coeff1])
            light_2_map = cv.merge([light_2[0]*coeff2, light_2[1]*coeff2, light_2[2]*coeff2])
            illuminant__map = light_1_map + light_2_map
        elif len(self.lights) == 3:
            coeff1, coeff2, coeff3 = cv.split(self.coeff_map)
            light_1 = np.array(self.lights[0], dtype=np.float32)
            light_2 = np.array(self.lights[1], dtype=np.float32)
            light_3 = np.array(self.lights[2], dtype=np.float32)
            light_1_map = cv.merge([light_1[0]*coeff1, light_1[1]*coeff1, light_1[2]*coeff1])
            light_2_map = cv.merge([light_2[0]*coeff2, light_2[1]*coeff2, light_2[2]*coeff2])
            light_3_map = cv.merge([light_3[0]*coeff3, light_3[1]*coeff3, light_3[2]*coeff3])
            illuminant__map = light_1_map + light_2_map + light_3_map

        illuminant__map = illuminant__map[..., ::-1]
            
        self.gt_image = self.image / illuminant__map
        self.gt_image = np.clip(self.gt_image, 0.0, 1.0)
    
    def get_image_name(self):
        return self.image_path.lower().split("/")[-1].split(".")[0]

    def get_image(self):
        return self.image
    
    def get_gt_image(self):
        return self.gt_image

    def get_info(self):
        return {
            "camera_model": self.camera_model,
            "place": self.place,
            "lights": self.lights,
            "coeff": self.coeff_map
        }