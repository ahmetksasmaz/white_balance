import cv2 as cv
import numpy as np
import scipy.io
import os
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *

class GehlerDataProvider(DataProvider):
    def __init__(self, saturation_mask=None, color_checker="patch", override_dimensions=(-1, -1)):
        # saturation_mask = (type, scope, threshold) or None
        # type = "raw" or "normalized"
        # scope = "all" or "any"
        # threshold = percentage
        super().__init__(override_dimensions)
        self.saturation_mask = saturation_mask
        self.color_checker = color_checker
        
        self.data_names = []
        self.cc_coords = []
        
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

        # Load Checker Coordinates
        coords_dir = os.path.join(ROOT_DIRECTORY, "groundtruth_568", "coordinates")
        for img_path in self.data_names:
            img_name = os.path.basename(img_path).replace("." + IMAGE_EXTENSION, "")
            txt_path = os.path.join(coords_dir, f"{img_name}_macbeth.txt")
            
            def load_gehler_coords(txt_p):
                if not os.path.exists(txt_p): return None
                with open(txt_p, 'r') as f:
                    lines = [l.strip().split() for l in f.readlines() if l.strip()]
                if not lines: return None
                
                # First line: reference dimensions (usually [Width, Height])
                ref_size = [float(x) for x in lines[0]]
                
                if len(lines) < 101: return None
                
                # Full checker corners: lines 2-5 (indices 1-4)
                # Matlab code uses order [2 4 5 3] corresponding to list indices [0, 2, 3, 1]
                l = lines[1:5]
                full_checker = [
                    [float(l[0][0]), float(l[0][1])], # Matlab 2 (index 1)
                    [float(l[2][0]), float(l[2][1])], # Matlab 4 (index 3)
                    [float(l[3][0]), float(l[3][1])], # Matlab 5 (index 4)
                    [float(l[1][0]), float(l[1][1])],  # Matlab 3 (index 2)
                ]
                
                # Patch corners: lines 6-101 (indices 5-100)
                patches = []
                for p in range(24):
                    # Applying same reordering for each patch as for the full checker
                    p_l = lines[5 + 4*p : 5 + 4*p + 4]
                    patch_pts = [
                        [float(p_l[0][0]), float(p_l[0][1])],
                        [float(p_l[1][0]), float(p_l[1][1])],
                        [float(p_l[2][0]), float(p_l[2][1])],
                        [float(p_l[3][0]), float(p_l[3][1])],
                    ]
                    patches.append(patch_pts)
                
                return {"ref_size": ref_size, "all": full_checker, "patch": patches}
            
            self.cc_coords.append(load_gehler_coords(txt_path))

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

        black_level = BLACK_LEVEL_1D if "canon1d" in image_path else BLACK_LEVEL_5D
        normalized_raw_image = np.clip((raw_image - black_level) / (SATURATION_LEVEL - black_level), 0, 1)
        data.set_quantization(SATURATION_LEVEL - black_level)

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

        # Load GT info
        illuminants = {}
        if self.real_rgb is not None and index < len(self.real_rgb):
            ill_rgb = self.real_rgb[index]
            ill_chroma = ill_rgb / np.linalg.norm(ill_rgb)
            rg = ill_chroma[0] / ill_chroma[1]
            bg = ill_chroma[2] / ill_chroma[1]
            illuminants["Illuminant1"] = (rg, bg)
        data.set_illuminants(illuminants)

        # Set checkerboard and saturation mask
        cc_data = self.cc_coords[index]
        mask_orig = None
        if cc_data is not None:
            mask_orig = np.ones((h_orig, w_orig), dtype=np.uint8)
            
            # Calculate scale factor based on Matlab snippet:
            # scale = cc_coord(1,[2 1])./[size(input_im,1) size(input_im,2)];
            # cc_coord(1,1) is Width, cc_coord(1,2) is Height
            ref_w, ref_h = cc_data["ref_size"]
            scale = [ref_h / h_orig, ref_w / w_orig]
            
            # Helper to scale points
            def scale_pts(pts_list):
                return [[p[0] / scale[0], p[1] / scale[1]] for p in pts_list]

            if self.color_checker == "patch":
                for patch_pts in cc_data["patch"]:
                    scaled_patch = scale_pts(patch_pts)
                    pts = np.array([scaled_patch], dtype=np.int32)
                    cv.fillPoly(mask_orig, pts, 0)
            elif self.color_checker == "all":
                scaled_checker = scale_pts(cc_data["all"])
                pts = np.array([scaled_checker], dtype=np.int32)
                cv.fillPoly(mask_orig, pts, 0)
        elif self.saturation_mask is not None:
            mask_orig = np.ones((h_orig, w_orig), dtype=np.uint8)

        if mask_orig is not None:
            if self.saturation_mask is not None:
                if self.saturation_mask[0] == "raw":
                    if self.saturation_mask[1] == "all":
                        mask_orig = mask_orig & np.all(raw_image <= SATURATION_LEVEL * self.saturation_mask[2], axis=2).astype(np.uint8)
                    elif self.saturation_mask[1] == "any":
                        mask_orig = mask_orig & np.any(raw_image <= SATURATION_LEVEL * self.saturation_mask[2], axis=2).astype(np.uint8)
                elif self.saturation_mask[0] == "normalized":
                    if self.saturation_mask[1] == "all":
                        mask_orig = mask_orig & np.all(normalized_raw_image <= self.saturation_mask[2], axis=2).astype(np.uint8)
                    elif self.saturation_mask[1] == "any":
                        mask_orig = mask_orig & np.any(normalized_raw_image <= self.saturation_mask[2], axis=2).astype(np.uint8)

            # Resize mask if image was resized
            if new_width > 0 or new_height > 0:
                mask = cv.resize(mask_orig, (new_width, new_height), interpolation=cv.INTER_NEAREST)
            else:
                mask = mask_orig

            # gamma_image = (np.power(normalized_raw_image, 1/2.2) * 255.0).astype(np.uint8)
            # masked_gamma_image = cv.bitwise_and(gamma_image, gamma_image, mask=mask)
            # cv.imwrite(f"masked_gamma_images/{data.get_image_name()}_masked_gamma.png", masked_gamma_image)

            data.set_mask(mask.astype(bool))

        return data

    def __len__(self):
        return len(self.data_names)