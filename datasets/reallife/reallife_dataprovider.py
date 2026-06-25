import cv2 as cv
import numpy as np
import os
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *


class ReallifeDataProvider(DataProvider):
    def __init__(self, override_dimensions=(-1, -1)):
        super().__init__(override_dimensions)
        self.data_names = IMAGE_NAMES

    def _construct_data(self, index):
        data = Data()
        image_name = self.data_names[index]
        data.set_image_name(image_name)

        raw_path = os.path.join(ROOT_DIRECTORY, f"{image_name}{LINEAR_CORRUPTED_SUFFIX}.{LINEAR_CORRUPTED_EXTENSION}")
        raw_image = cv.imread(raw_path, cv.IMREAD_UNCHANGED)
        if raw_image is None:
            raise ValueError(f"Failed to load rawlike image for {image_name} from {raw_path}")

        raw_image = np.clip(raw_image.astype(np.float32) / 255.0, 0.0, 1.0)
        data.set_quantization(255)

        h_orig, w_orig = raw_image.shape[:2]
        if self.override_dimensions[0] > 0 and self.override_dimensions[1] > 0:
            raw_image = cv.resize(raw_image, (self.override_dimensions[0], self.override_dimensions[1]), interpolation=cv.INTER_AREA)
        elif self.override_dimensions[0] > 0:
            new_width = self.override_dimensions[0]
            raw_image = cv.resize(raw_image, (new_width, int(new_width / (w_orig / h_orig))), interpolation=cv.INTER_AREA)
        elif self.override_dimensions[1] > 0:
            new_height = self.override_dimensions[1]
            raw_image = cv.resize(raw_image, (int(new_height * (w_orig / h_orig)), new_height), interpolation=cv.INTER_AREA)

        data.set_raw_image(raw_image)

        srgb_path = os.path.join(ROOT_DIRECTORY, f"{image_name}.{SRGB_EXTENSION}")
        srgb_image = cv.imread(srgb_path, cv.IMREAD_UNCHANGED)
        if srgb_image is None:
            raise ValueError(f"Failed to load sRGB image for {image_name} from {srgb_path}")

        data.set_srgb_image(srgb_image)
        data.set_mask(None)
        data.set_illuminants({})
        return data
