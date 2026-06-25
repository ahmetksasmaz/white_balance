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
            for line in f.readlines()[1:]:
                line = line.strip()
                i = line.index(",")
                self.gt_lines[self.data_names.index(line[:i])] = line[i+1:]

        with open(ROOT_DIRECTORY+"/properties.csv", "r") as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                i = line.index(",")
                self.properties_lines[self.data_names.index(line[:i])] = line[i+1:]

    def _construct_data(self, index):
        data = Data()
        data.set_image_name(self.data_names[index])

        image_path = INPUT_IMAGE_DIRECTORY + "/" + self.data_names[index] + "." + IMAGE_EXTENSION
        raw_image = cv.imread(image_path, cv.IMREAD_UNCHANGED).astype(np.float32)
        raw_image = np.clip((raw_image - BLACK_LEVEL) / (SATURATION_LEVEL - BLACK_LEVEL), 0, 1)
        data.set_quantization(SATURATION_LEVEL - BLACK_LEVEL)

        if self.override_dimensions[0] > 0 and self.override_dimensions[1] > 0:
            raw_image = cv.resize(raw_image, self.override_dimensions)
        elif self.override_dimensions[0] > 0:
            new_width = self.override_dimensions[0]
            raw_image = cv.resize(raw_image, (new_width, int(new_width / (raw_image.shape[1] / raw_image.shape[0]))))
        elif self.override_dimensions[1] > 0:
            new_height = self.override_dimensions[1]
            raw_image = cv.resize(raw_image, (int(new_height * (raw_image.shape[1] / raw_image.shape[0])), new_height))
        data.set_raw_image(raw_image)

        gt_infos = [float(x) if x else None for x in self.gt_lines[index].split(",")]
        mean_rgb = gt_infos[0:3][::-1]
        left_rgb = gt_infos[3:6][::-1]
        right_rgb = gt_infos[6:9][::-1]
        left_white_rgb = gt_infos[9:12][::-1]
        right_white_rgb = gt_infos[12:15][::-1]

        def to_chroma(rgb):
            if rgb[0] is None: return None
            c = np.array(rgb) / np.linalg.norm(rgb)
            return (c[2] / c[1], c[0] / c[1])

        data.set_illuminants({
            "mean": to_chroma(mean_rgb),
            "left": to_chroma(left_rgb),
            "right": to_chroma(right_rgb),
            "left_white": to_chroma(left_white_rgb),
            "right_white": to_chroma(right_white_rgb),
        })

        props = self.properties_lines[index].split(",")
        iso = float(props[18]) if props[18] else None
        aperture = float(props[19]) if props[19] else None
        exposure_time = float(props[20]) if props[20] else None
        data.set_exposure_values(exposure_time, iso, aperture)

        return data
