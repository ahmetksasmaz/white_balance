import cv2 as cv
import numpy as np
from .configuration import BLACK_LEVEL, SATURATION_LEVEL
from chromatic_adaptation import *

class CubePPData:
    def __init__(self, image_path : str, gt_info : str, load_resized=False):
        self.image_path = image_path
        self.gt_info = gt_info
        self.image = None
        self.gt_image = None
        self.load_resized = load_resized

        self.__parse_gt_info()
        self.__load_image()
        self.__construct_gt_image()

    def __parse_gt_info(self):
        # Parse the ground truth information
        gt_info_splitted = self.gt_info.split(",")
        gt_infos = [float(x) if x else None for x in gt_info_splitted]
        self.mean_rgb = gt_infos[0:3]
        self.left_rgb = gt_infos[3:6]
        self.right_rgb = gt_infos[6:9]
        self.left_white_rgb = gt_infos[9:12]
        self.right_white_rgb = gt_infos[12:15]

    def __load_image(self):
        # Load the image using OpenCV
        self.image = cv.imread(self.image_path, cv.IMREAD_UNCHANGED)
        self.image = self.image.astype(np.float32)

        self.image = np.clip((self.image - BLACK_LEVEL) / (SATURATION_LEVEL - BLACK_LEVEL), 0, 1)

        if self.load_resized:
            orig_h, orig_w = self.image.shape[:2]
            new_w = int(orig_w * (512 / orig_h))
            self.image = cv.resize(self.image, (new_w, 512))

    def __construct_gt_image(self):
        if self.mean_rgb[0] is not None:
            mean_rgb = np.array(self.mean_rgb[::-1]) # RGB to BGR order
            normalized_gt = mean_rgb / np.linalg.norm(mean_rgb)
            ground_truth_illuminant = WhitePoint(normalized_gt, "SRGB")
            gt_target_illuminant = D65_WHITE_POINT
            gt_target_illuminant.adjust_luminance(ground_truth_illuminant.luminance())

            self.gt_image = chromatic_adaptation(self.image, ground_truth_illuminant, gt_target_illuminant)

            if self.load_resized:
                orig_h, orig_w = self.gt_image.shape[:2]
                new_w = int(orig_w * (512 / orig_h))
                self.gt_image = cv.resize(self.gt_image, (new_w, 512))

    def get_image_name(self):
        return self.image_path.lower().split("/")[-1].split(".")[0]

    def get_image(self):
        return self.image
    
    def get_gt_image(self):
        return self.gt_image
    
    def get_info(self):
        return {
            "mean_rgb": self.mean_rgb,
            "left_rgb": self.left_rgb,
            "right_rgb": self.right_rgb,
            "left_white_rgb": self.left_white_rgb,
            "right_white_rgb": self.right_white_rgb
        }