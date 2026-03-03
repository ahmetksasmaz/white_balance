import cv2 as cv
import numpy as np
import scipy.io
import os
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *

class NUS8DataProvider(DataProvider):
    EXCLUDED_CAMERAS = ["NikonD40"]

    def __init__(self, override_dimensions=(-1, -1)):
        super().__init__(override_dimensions)
        
        self.data_names = []
        self.gt_illuminants = []
        self.darkness_levels = []
        self.saturation_levels = []
        self.cc_coords = []
        
        # Iterating over the 8 camera subdirectories
        if os.path.exists(ROOT_DIRECTORY):
            camera_dirs = sorted([d for d in os.listdir(ROOT_DIRECTORY) if os.path.isdir(os.path.join(ROOT_DIRECTORY, d)) and not d.startswith('.') and d not in self.EXCLUDED_CAMERAS])
            for cam_dir in camera_dirs:
                cam_path = os.path.join(ROOT_DIRECTORY, cam_dir)
                gt_mat_path = os.path.join(cam_path, f"{cam_dir}_gt.mat")
                
                if os.path.exists(gt_mat_path):
                    mat_data = scipy.io.loadmat(gt_mat_path)
                    
                    # Extract variables from the mat file
                    darkness_level = float(mat_data['darkness_level'][0][0])
                    saturation_level = float(mat_data['saturation_level'][0][0])
                    all_image_names = [name[0][0] for name in mat_data['all_image_names']]
                    groundtruth_illuminants = mat_data['groundtruth_illuminants']
                    cc_coords = mat_data['CC_coords'] if 'CC_coords' in mat_data else None
                    
                    # Ensure PNG images exist and append properties
                    for i, img_name in enumerate(all_image_names):
                        img_path = os.path.join(cam_path, "PNG", f"{img_name}.{IMAGE_EXTENSION}")
                        if os.path.exists(img_path):
                            self.data_names.append(img_path)
                            self.darkness_levels.append(darkness_level)
                            self.saturation_levels.append(saturation_level)
                            self.gt_illuminants.append(groundtruth_illuminants[i])
                            self.cc_coords.append(cc_coords[i] if cc_coords is not None else None)
                        else:
                            # Lowercase extension check just in case
                            img_path_lower = os.path.join(cam_path, "PNG", f"{img_name}.{IMAGE_EXTENSION.lower()}")
                            if os.path.exists(img_path_lower):
                                self.data_names.append(img_path_lower)
                                self.darkness_levels.append(darkness_level)
                                self.saturation_levels.append(saturation_level)
                                self.gt_illuminants.append(groundtruth_illuminants[i])
                                self.cc_coords.append(cc_coords[i] if cc_coords is not None else None)


    def _construct_data(self, index):
        data = Data()
        image_path = self.data_names[index]
        image_name = os.path.basename(image_path).split('.')[0]
        data.set_image_name(image_name)

        # Load Raw Image
        raw_image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if raw_image is None:
            raise ValueError(f"Failed to load image at {image_path}")
            
        raw_image = raw_image.astype(np.float32)

        # Apply camera-specific levels
        black_level = self.darkness_levels[index]
        saturation_level = self.saturation_levels[index]

        raw_image = np.clip((raw_image - black_level) / (saturation_level - black_level), 0, 1)
        data.set_quantization(saturation_level - black_level)

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

        # Load GT info -> Store as a dict
        illuminants = {}
        if index < len(self.gt_illuminants):
            ill_rgb = self.gt_illuminants[index]
            # Convert to chroma (r/g, b/g)
            ill_chroma = ill_rgb / np.linalg.norm(ill_rgb)
            rg = ill_chroma[0] / ill_chroma[1]
            bg = ill_chroma[2] / ill_chroma[1]
            illuminants["Illuminant1"] = (rg, bg)
            
        data.set_illuminants(illuminants)

        # Set checkerboard mask from CC_coords [row_start, row_end, col_start, col_end]
        cc = self.cc_coords[index]
        if cc is not None:
            h, w = raw_image.shape[:2]
            mask = np.ones((h, w), dtype=bool)
            r_start, r_end, c_start, c_end = int(cc[0]), int(cc[1]), int(cc[2]), int(cc[3])
            # Clamp to image bounds
            r_start = max(0, r_start)
            r_end = min(h, r_end)
            c_start = max(0, c_start)
            c_end = min(w, c_end)
            mask[r_start:r_end, c_start:c_end] = False
            data.set_mask(mask)

        return data

    def __len__(self):
        return len(self.data_names)
