import cv2 as cv
import numpy as np
import scipy.io
import os
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *

class GehlerDataProvider(DataProvider):
    def __init__(self, override_dimensions=(-1, -1)):
        super().__init__(override_dimensions)
        
        self.data_names = []
        
        # Load all 1D images
        canon1d_dir = os.path.join(ROOT_DIRECTORY, "canon1d")
        if os.path.exists(canon1d_dir):
            canon1d_files = sorted([f for f in os.listdir(canon1d_dir) if f.endswith(IMAGE_EXTENSION)])
            for f in canon1d_files:
                self.data_names.append(os.path.join(canon1d_dir, f))
                
        # Load all 5D images
        canon5d_dir = os.path.join(ROOT_DIRECTORY, "canon5d")
        if os.path.exists(canon5d_dir):
            canon5d_files = sorted([f for f in os.listdir(canon5d_dir) if f.endswith(IMAGE_EXTENSION)])
            for f in canon5d_files:
                self.data_names.append(os.path.join(canon5d_dir, f))
        
        # Ensure we found exactly 568 images (as dataset is referred to as "568")
        # Load Ground Truth
        mat_path = os.path.join(ROOT_DIRECTORY, "groundtruth_568", "real_illum_568..mat")
        if os.path.exists(mat_path):
            mat_data = scipy.io.loadmat(mat_path)
            self.real_rgb = mat_data['real_rgb']
        else:
            self.real_rgb = None

    def _construct_data(self, index):
        data = Data()
        image_path = self.data_names[index]
        image_name = os.path.basename(image_path).replace("." + IMAGE_EXTENSION, "")
        data.set_image_name(image_name)

        # Load Raw Image
        raw_image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if raw_image is None:
            raise ValueError(f"Failed to load image at {image_path}")
            
        raw_image = raw_image.astype(np.float32)

        # Apply specific black level
        if "canon1d" in image_path:
            black_level = BLACK_LEVEL_1D
        else:
            black_level = BLACK_LEVEL_5D

        raw_image = np.clip((raw_image - black_level) / (SATURATION_LEVEL - black_level), 0, 1)
        data.set_quantization(SATURATION_LEVEL - black_level)

        # Override dimensions if specified
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

        # Load GT info
        illuminants = {}
        if self.real_rgb is not None and index < len(self.real_rgb):
            ill_rgb = self.real_rgb[index]
            # Convert to chroma (r/g, b/g)
            ill_chroma = ill_rgb / np.linalg.norm(ill_rgb)
            rg = ill_chroma[0] / ill_chroma[1]  # 'real_rgb' channel order is standard RGB
            bg = ill_chroma[2] / ill_chroma[1]
            illuminants["Illuminant1"] = (rg, bg)
            
        data.set_illuminants(illuminants)

        return data

    def __len__(self):
        return len(self.data_names)