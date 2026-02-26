import cv2 as cv
import numpy as np
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *

class CubePPDataProvider(DataProvider):
    def __init__(self, override_dimensions=(-1, -1)):
        super().__init__(override_dimensions)
        
        self.data_names = IMAGE_LIST
        self.gt_lines = [""] * len(self.data_names)
        self.properties_lines = [""] * len(self.data_names)

        with open(ROOT_DIRECTORY+"/gt.csv", "r") as f:
            raw_lines = f.readlines()
            for line in raw_lines[1:]:
                line = line.strip()
                first_comma_index = line.index(",")
                image_name = line[:first_comma_index]
                self.gt_lines[self.data_names.index(image_name)] = line[first_comma_index+1:]
        
        with open(ROOT_DIRECTORY+"/properties.csv", "r") as f:
            raw_lines = f.readlines()
            for line in raw_lines[1:]:
                line = line.strip()
                first_comma_index = line.index(",")
                image_name = line[:first_comma_index]
                self.properties_lines[self.data_names.index(image_name)] = line[first_comma_index+1:]

    def _construct_data(self, index):
        data = Data()
        data.set_image_name(self.data_names[index])

        # Load Raw Image
        image_path = INPUT_IMAGE_DIRECTORY + "/" + self.data_names[index] + "." + IMAGE_EXTENSION
        raw_image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        raw_image = raw_image.astype(np.float32)
        raw_image = np.clip((raw_image - BLACK_LEVEL) / (SATURATION_LEVEL - BLACK_LEVEL), 0, 1)
        
        # Override dimensions if specified
        if self.override_dimensions[0] > 0 and self.override_dimensions[1] > 0:
            raw_image = cv.resize(raw_image, self.override_dimensions)
        elif self.override_dimensions[0] > 0:
            aspect_ratio = raw_image.shape[1] / raw_image.shape[0]
            new_width = self.override_dimensions[0]
            new_height = int(new_width / aspect_ratio)
            raw_image = cv.resize(raw_image, (new_width, new_height))
        elif self.override_dimensions[1] > 0:
            aspect_ratio = raw_image.shape[1] / raw_image.shape[0]
            new_height = self.override_dimensions[1]
            new_width = int(new_height * aspect_ratio)
            raw_image = cv.resize(raw_image, (new_width, new_height))
        
        data.set_raw_image(raw_image)

        # Load GT info
        gt_info_splitted = self.gt_lines[index].split(",")
        gt_infos = [float(x) if x else None for x in gt_info_splitted]
        mean_rgb = gt_infos[0:3][::-1]
        left_rgb = gt_infos[3:6][::-1]
        right_rgb = gt_infos[6:9][::-1]
        left_white_rgb = gt_infos[9:12][::-1]
        right_white_rgb = gt_infos[12:15][::-1]
        illuminants = {
            "mean": None,
            "left": None,
            "right": None,
            "left_white": None,
            "right_white": None
        }
        if mean_rgb[0] is not None:
            mean_chroma = np.array(mean_rgb) / np.linalg.norm(mean_rgb)
            rg = mean_chroma[2] / mean_chroma[1]
            bg = mean_chroma[0] / mean_chroma[1]
            illuminants["mean"] = (rg, bg)
        if left_rgb[0] is not None:
            left_chroma = np.array(left_rgb) / np.linalg.norm(left_rgb)
            rg = left_chroma[2] / left_chroma[1]
            bg = left_chroma[0] / left_chroma[1]
            illuminants["left"] = (rg, bg)
        if right_rgb[0] is not None:
            right_chroma = np.array(right_rgb) / np.linalg.norm(right_rgb)
            rg = right_chroma[2] / right_chroma[1]
            bg = right_chroma[0] / right_chroma[1]
            illuminants["right"] = (rg, bg)
        if left_white_rgb[0] is not None:
            left_white_chroma = np.array(left_white_rgb) / np.linalg.norm(left_white_rgb)
            rg = left_white_chroma[2] / left_white_chroma[1]
            bg = left_white_chroma[0] / left_white_chroma[1]
            illuminants["left_white"] = (rg, bg)
        if right_white_rgb[0] is not None:
            right_white_chroma = np.array(right_white_rgb) / np.linalg.norm(right_white_rgb)
            rg = right_white_chroma[2] / right_white_chroma[1]
            bg = right_white_chroma[0] / right_white_chroma[1]
            illuminants["right_white"] = (rg, bg)
        data.set_illuminants(illuminants)

        properties_info_splitted = self.properties_lines[index].split(",")
        # Load Scene Data

        # Not necessary for now, but can be added later if needed

        # Load Exposure Values
        iso = float(properties_info_splitted[18]) if properties_info_splitted[18] else None
        aperture = float(properties_info_splitted[19]) if properties_info_splitted[19] else None
        exposure_time = float(properties_info_splitted[20]) if properties_info_splitted[20] else None
        data.set_exposure_values(exposure_time, iso, aperture)

        return data