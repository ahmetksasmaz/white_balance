import cv2 as cv
import numpy as np
import scipy.io
import os
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *

class NUS8DataProvider(DataProvider):
    EXCLUDED_CAMERAS = ["NikonD40"]

    def __init__(self, saturation_mask=('raw', 'all', 0.98), color_checker="patch", override_dimensions=(-1, -1)):
        super().__init__(override_dimensions)

        self.data_names = []
        self.gt_illuminants = []
        self.darkness_levels = []
        self.saturation_levels = []
        self.cc_coords = []
        self.saturation_mask = saturation_mask
        self.color_checker = color_checker

        if os.path.exists(ROOT_DIRECTORY):
            camera_dirs = sorted([d for d in os.listdir(ROOT_DIRECTORY) if os.path.isdir(os.path.join(ROOT_DIRECTORY, d)) and not d.startswith('.') and d not in self.EXCLUDED_CAMERAS])
            for cam_dir in camera_dirs:
                cam_path = os.path.join(ROOT_DIRECTORY, cam_dir)
                gt_mat_path = os.path.join(cam_path, f"{cam_dir}_gt.mat")

                if os.path.exists(gt_mat_path):
                    mat_data = scipy.io.loadmat(gt_mat_path)
                    darkness_level = float(mat_data['darkness_level'][0][0])
                    saturation_level = float(mat_data['saturation_level'][0][0])
                    all_image_names = [name[0][0] for name in mat_data['all_image_names']]
                    groundtruth_illuminants = mat_data['groundtruth_illuminants']

                    for i, img_name in enumerate(all_image_names):
                        img_path = os.path.join(cam_path, "PNG", f"{img_name}.{IMAGE_EXTENSION}")
                        txt_path = os.path.join(cam_path, "CHECKER", f"{img_name}_mask.txt")

                        def load_mask_coords(txt_p):
                            if not os.path.exists(txt_p): return None
                            patches = []
                            with open(txt_p, 'r') as f:
                                lines = [l.strip() for l in f.readlines() if l.strip()]
                            if len(lines) >= 49:
                                roi_line = lines[0].split(',')
                                roi_x, roi_y = float(roi_line[0]), float(roi_line[1])
                                for p in range(24):
                                    xm = [float(v) for v in lines[1 + 2*p].split(',')]
                                    ym = [float(v) for v in lines[2 + 2*p].split(',')]
                                    patches.append(([x + roi_x for x in xm], [y + roi_y for y in ym]))
                            return patches if patches else None

                        img_path_lower = os.path.join(cam_path, "PNG", f"{img_name}.{IMAGE_EXTENSION.lower()}")
                        actual_img_path = img_path if os.path.exists(img_path) else (img_path_lower if os.path.exists(img_path_lower) else None)

                        if actual_img_path:
                            self.data_names.append(actual_img_path)
                            self.darkness_levels.append(darkness_level)
                            self.saturation_levels.append(saturation_level)
                            self.gt_illuminants.append(groundtruth_illuminants[i])
                            self.cc_coords.append(load_mask_coords(txt_path))

    def _construct_data(self, index):
        data = Data()
        image_path = self.data_names[index]
        data.set_image_name(os.path.basename(image_path).split('.')[0])

        raw_image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if raw_image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        raw_image = raw_image.astype(np.float32)

        black_level = self.darkness_levels[index]
        saturation_level = self.saturation_levels[index]
        normalized_raw_image = np.clip((raw_image - black_level) / (saturation_level - black_level), 0, 1)
        data.set_quantization(saturation_level - black_level)

        h_orig, w_orig = normalized_raw_image.shape[:2]
        new_width, new_height = -1, -1
        if self.override_dimensions[0] > 0 and self.override_dimensions[1] > 0:
            new_width, new_height = self.override_dimensions[0], self.override_dimensions[1]
            normalized_raw_image = cv.resize(normalized_raw_image, (new_width, new_height))
        elif self.override_dimensions[0] > 0:
            new_width = self.override_dimensions[0]
            new_height = int(new_width / (w_orig / h_orig))
            normalized_raw_image = cv.resize(normalized_raw_image, (new_width, new_height))
        elif self.override_dimensions[1] > 0:
            new_height = self.override_dimensions[1]
            new_width = int(new_height * (w_orig / h_orig))
            normalized_raw_image = cv.resize(normalized_raw_image, (new_width, new_height))
        data.set_raw_image(normalized_raw_image)

        illuminants = {}
        if index < len(self.gt_illuminants):
            ill_rgb = self.gt_illuminants[index]
            ill_chroma = ill_rgb / np.linalg.norm(ill_rgb)
            illuminants["Illuminant1"] = (ill_chroma[0] / ill_chroma[1], ill_chroma[2] / ill_chroma[1])
        data.set_illuminants(illuminants)

        cc_coord = self.cc_coords[index]
        if cc_coord is not None:
            mask_orig = np.ones((h_orig, w_orig), dtype=np.uint8)

            if self.color_checker == "patches":
                for (xm, ym) in cc_coord:
                    cv.fillPoly(mask_orig, np.array([[[x, y] for x, y in zip(xm, ym)]], dtype=np.int32), 0)
            elif self.color_checker == "all":
                x_min = int(min([min(xm) for xm, ym in cc_coord]))
                x_max = int(max([max(xm) for xm, ym in cc_coord]))
                y_min = int(min([min(ym) for xm, ym in cc_coord]))
                y_max = int(max([max(ym) for xm, ym in cc_coord]))
                mask_orig[y_min:y_max, x_min:x_max] = 0

            if self.saturation_mask is not None:
                if self.saturation_mask[0] == "raw":
                    if self.saturation_mask[1] == "all":
                        mask_orig = mask_orig & np.all(raw_image <= saturation_level * self.saturation_mask[2], axis=2).astype(np.uint8)
                    elif self.saturation_mask[1] == "any":
                        mask_orig = mask_orig & np.any(raw_image <= saturation_level * self.saturation_mask[2], axis=2).astype(np.uint8)
                elif self.saturation_mask[0] == "normalized":
                    if self.saturation_mask[1] == "all":
                        mask_orig = mask_orig & np.all(normalized_raw_image <= self.saturation_mask[2], axis=2).astype(np.uint8)
                    elif self.saturation_mask[1] == "any":
                        mask_orig = mask_orig & np.any(normalized_raw_image <= self.saturation_mask[2], axis=2).astype(np.uint8)

            mask = cv.resize(mask_orig, (new_width, new_height), interpolation=cv.INTER_NEAREST) if (new_width > 0 or new_height > 0) else mask_orig
            data.set_mask(mask.astype(bool))

        return data
