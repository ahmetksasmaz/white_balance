import cv2 as cv
import numpy as np
import scipy.io
import os
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *

class GehlerDataProvider(DataProvider):
    def __init__(self, saturation_mask=None, color_checker="patch", override_dimensions=(-1, -1)):
        super().__init__(override_dimensions)
        self.saturation_mask = saturation_mask
        self.color_checker = color_checker

        self.data_names = []
        self.cc_coords = []

        canon1d_dir = os.path.join(ROOT_DIRECTORY, "canon1d")
        if os.path.exists(canon1d_dir):
            for f in sorted([f for f in os.listdir(canon1d_dir) if f.endswith(IMAGE_EXTENSION)]):
                self.data_names.append(os.path.join(canon1d_dir, f))

        canon5d_dir = os.path.join(ROOT_DIRECTORY, "canon5d")
        if os.path.exists(canon5d_dir):
            for f in sorted([f for f in os.listdir(canon5d_dir) if f.endswith(IMAGE_EXTENSION)]):
                self.data_names.append(os.path.join(canon5d_dir, f))

        mat_path = os.path.join(ROOT_DIRECTORY, "groundtruth_568", "real_illum_568..mat")
        if os.path.exists(mat_path):
            self.real_rgb = scipy.io.loadmat(mat_path)['real_rgb']
        else:
            self.real_rgb = None

        coords_dir = os.path.join(ROOT_DIRECTORY, "groundtruth_568", "coordinates")
        for img_path in self.data_names:
            img_name = os.path.basename(img_path).replace("." + IMAGE_EXTENSION, "")
            txt_path = os.path.join(coords_dir, f"{img_name}_macbeth.txt")

            def load_gehler_coords(txt_p):
                if not os.path.exists(txt_p): return None
                with open(txt_p, 'r') as f:
                    lines = [l.strip().split() for l in f.readlines() if l.strip()]
                if not lines or len(lines) < 101: return None
                ref_size = [float(x) for x in lines[0]]
                l = lines[1:5]
                full_checker = [
                    [float(l[0][0]), float(l[0][1])],
                    [float(l[2][0]), float(l[2][1])],
                    [float(l[3][0]), float(l[3][1])],
                    [float(l[1][0]), float(l[1][1])],
                ]
                patches = []
                for p in range(24):
                    p_l = lines[5 + 4*p : 5 + 4*p + 4]
                    patches.append([
                        [float(p_l[0][0]), float(p_l[0][1])],
                        [float(p_l[1][0]), float(p_l[1][1])],
                        [float(p_l[2][0]), float(p_l[2][1])],
                        [float(p_l[3][0]), float(p_l[3][1])],
                    ])
                return {"ref_size": ref_size, "all": full_checker, "patch": patches}

            self.cc_coords.append(load_gehler_coords(txt_path))

    def get_image_name(self, index):
        image_path = self.data_names[index]
        return os.path.basename(image_path).replace("." + IMAGE_EXTENSION, "")

    def _construct_data(self, index):
        data = Data()
        image_path = self.data_names[index]
        image_name = self.get_image_name(index)
        data.set_image_name(image_name)

        raw_image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if raw_image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        raw_image = raw_image.astype(np.float32)

        black_level = BLACK_LEVEL_1D if "canon1d" in image_path else BLACK_LEVEL_5D
        raw_image = raw_image - black_level
        normalized_raw_image = np.clip(raw_image / raw_image.max(), 0, 1)
        data.set_quantization(SATURATION_LEVEL - black_level)

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
        if self.real_rgb is not None and index < len(self.real_rgb):
            ill_rgb = self.real_rgb[index]
            ill_chroma = ill_rgb / np.linalg.norm(ill_rgb)
            illuminants["Illuminant1"] = (ill_chroma[0] / ill_chroma[1], ill_chroma[2] / ill_chroma[1])
        data.set_illuminants(illuminants)

        cc_data = self.cc_coords[index]
        mask_orig = None
        if cc_data is not None:
            mask_orig = np.ones((h_orig, w_orig), dtype=np.uint8)
            ref_w, ref_h = cc_data["ref_size"]
            scale = [ref_h / h_orig, ref_w / w_orig]

            def scale_pts(pts_list):
                return [[p[0] / scale[0], p[1] / scale[1]] for p in pts_list]

            if self.color_checker == "patch":
                for patch_pts in cc_data["patch"]:
                    cv.fillPoly(mask_orig, np.array([scale_pts(patch_pts)], dtype=np.int32), 0)
            elif self.color_checker == "all":
                cv.fillPoly(mask_orig, np.array([scale_pts(cc_data["all"])], dtype=np.int32), 0)
        elif self.saturation_mask is not None:
            mask_orig = np.ones((h_orig, w_orig), dtype=np.uint8)

        if mask_orig is not None:
            if self.saturation_mask is not None:
                sat_mask = np.ones((h_orig, w_orig), dtype=np.uint8)
                if self.saturation_mask[0] == "raw":
                    if self.saturation_mask[1] == "all":
                        sat_mask = np.all(raw_image <= raw_image.max() * self.saturation_mask[2], axis=2).astype(np.uint8)
                    elif self.saturation_mask[1] == "any":
                        sat_mask = np.any(raw_image <= raw_image.max() * self.saturation_mask[2], axis=2).astype(np.uint8)
                elif self.saturation_mask[0] == "normalized":
                    if self.saturation_mask[1] == "all":
                        sat_mask = np.all(normalized_raw_image <= self.saturation_mask[2], axis=2).astype(np.uint8)
                    elif self.saturation_mask[1] == "any":
                        sat_mask = np.any(normalized_raw_image <= self.saturation_mask[2], axis=2).astype(np.uint8)
                sat_mask = cv.dilate(sat_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
                mask_orig = mask_orig & sat_mask

            mask = cv.resize(mask_orig, (new_width, new_height), interpolation=cv.INTER_NEAREST) if (new_width > 0 or new_height > 0) else mask_orig
            data.set_mask(mask.astype(bool))

        return data
