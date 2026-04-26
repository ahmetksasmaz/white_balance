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
        self.root_dir = ROOT_DIRECTORY
        self.linear_corrupted_suffix = LINEAR_CORRUPTED_SUFFIX
        self.linear_corrupted_ext = LINEAR_CORRUPTED_EXTENSION
        self.srgb_ext = SRGB_EXTENSION

    def _construct_data(self, index):
        data = Data()
        image_name = self.data_names[index]
        data.set_image_name(image_name)

        raw_path = os.path.join(
            self.root_dir,
            f"{image_name}{self.linear_corrupted_suffix}.{self.linear_corrupted_ext}",
        )
        raw_image = cv.imread(raw_path, cv.IMREAD_UNCHANGED)
        if raw_image is None:
            raise ValueError(f"Failed to load rawlike image for {image_name} from {raw_path}")

        raw_image = raw_image.astype(np.float32)
        raw_image = raw_image / 255.0
        raw_image = np.clip(raw_image, 0.0, 1.0)
        data.set_quantization(255)

        h_orig, w_orig = raw_image.shape[:2]

        if self.override_dimensions[0] > 0 and self.override_dimensions[1] > 0:
            raw_image = cv.resize(raw_image, (self.override_dimensions[0], self.override_dimensions[1]), interpolation=cv.INTER_AREA)
        elif self.override_dimensions[0] > 0:
            aspect_ratio = w_orig / h_orig
            new_width = self.override_dimensions[0]
            new_height = int(new_width / aspect_ratio)
            raw_image = cv.resize(raw_image, (new_width, new_height), interpolation=cv.INTER_AREA)
        elif self.override_dimensions[1] > 0:
            aspect_ratio = w_orig / h_orig
            new_height = self.override_dimensions[1]
            new_width = int(new_height * aspect_ratio)
            raw_image = cv.resize(raw_image, (new_width, new_height), interpolation=cv.INTER_AREA)

        data.set_raw_image(raw_image)

        srgb_path = os.path.join(self.root_dir, f"{image_name}.{self.srgb_ext}")
        srgb_image = cv.imread(srgb_path, cv.IMREAD_UNCHANGED)
        if srgb_image is None:
            raise ValueError(f"Failed to load sRGB image for {image_name} from {srgb_path}")

        data.set_srgb_image(srgb_image)
        data.set_mask(None)
        data.set_illuminants({})
        return data

    def __len__(self):
        return len(self.data_names)
