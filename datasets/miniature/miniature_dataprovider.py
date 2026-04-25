import cv2 as cv
import numpy as np
import scipy.io
import os
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *

class MiniatureDataProvider(DataProvider):
    def __init__(self, override_dimensions=(-1, -1)):
        super().__init__(override_dimensions)
        self.data_names = IMAGE_NAMES
        self.linear_tiff_dir = os.path.join(ROOT_DIRECTORY, LINEAR_TIFF_IMAGE_DIRECTORY)
        self.srgb_dir = os.path.join(ROOT_DIRECTORY, SRGB_IMAGE_DIRECTORY)
        self.mask_dir = os.path.join(ROOT_DIRECTORY, MASK_DIRECTORY)
        self.metadata_dir = os.path.join(ROOT_DIRECTORY, METADATA_DIRECTORY)

    def _construct_data(self, index):
        data = Data()
        image_name = self.data_names[index]
        data.set_image_name(image_name)

        # Load Linear TIFF Image
        raw_image = cv.imread(os.path.join(self.linear_tiff_dir, f"{image_name}.{RAW_EXTENSION}"), cv.IMREAD_UNCHANGED)
        if raw_image is None:
            raise ValueError(f"Failed to load raw image for {image_name}")
            
        raw_image = raw_image.astype(np.float32)

        raw_image = raw_image - BLACK_LEVEL
        normalized_raw_image = raw_image / (SATURATION_LEVEL - BLACK_LEVEL)
        normalized_raw_image = np.clip(normalized_raw_image, 0, 1)
        data.set_quantization(SATURATION_LEVEL - BLACK_LEVEL)

        h_orig, w_orig = normalized_raw_image.shape[:2]

        # Override dimensions if specified
        new_width, new_height = -1, -1
        if self.override_dimensions[0] > 0 and self.override_dimensions[1] > 0:
            new_width, new_height = self.override_dimensions[0], self.override_dimensions[1]
            normalized_raw_image = cv.resize(normalized_raw_image, (new_width, new_height))
        elif self.override_dimensions[0] > 0:
            aspect_ratio = w_orig / h_orig
            new_width = self.override_dimensions[0]
            new_height = int(new_width / aspect_ratio)
            normalized_raw_image = cv.resize(normalized_raw_image, (new_width, new_height))
        elif self.override_dimensions[1] > 0:
            aspect_ratio = w_orig / h_orig
            new_height = self.override_dimensions[1]
            new_width = int(new_height * aspect_ratio)
            normalized_raw_image = cv.resize(normalized_raw_image, (new_width, new_height))
        
        data.set_raw_image(normalized_raw_image)

        # Load sRGB Image
        srgb_image = cv.imread(os.path.join(self.srgb_dir, f"{image_name}.{SRGB_EXTENSION}"), cv.IMREAD_UNCHANGED)
        if srgb_image is None:
            raise ValueError(f"Failed to load sRGB image for {image_name}")

        data.set_srgb_image(srgb_image)

        # Load mask if available
        mask_path = os.path.join(self.mask_dir, f"{image_name}_mask.{MASK_EXTENSION}")
        mask_image = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
        if mask_image is not None:
            if mask_image.ndim == 3 and mask_image.shape[2] == 4:
                alpha = mask_image[..., 3]
                rgb = mask_image[..., :3]
                mask_bool = (alpha != 0) & np.any(rgb != 0, axis=-1)
            elif mask_image.ndim == 3:
                mask_bool = np.any(mask_image != 0, axis=-1)
            else:
                mask_bool = mask_image != 0
            mask_bool = mask_bool.astype(np.uint8)
            mask_bool = cv.resize(mask_bool, (w_orig, h_orig), interpolation=cv.INTER_NEAREST).astype(bool)
            data.set_mask(mask_bool)
        else:
            data.set_mask(None)

        # Load GT info
        data.set_illuminants({})

        return data

    def __len__(self):
        return len(self.data_names)