import cv2 as cv
import numpy as np
from .configuration import BLACK_LEVEL, SATURATION_LEVEL

class CubePPData:
    def __init__(self, image_path : str, gt_info : str):
        self.image_path = image_path
        self.gt_info = gt_info

        self.__parse_gt_info()
        self.__load_image()

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

    def get_image_name(self):
        return self.image_path.lower().split("/")[-1].split(".")[0]

    def get_image(self):
        return self.image
    
    def get_info(self):
        return {
            "mean_rgb": self.mean_rgb,
            "left_rgb": self.left_rgb,
            "right_rgb": self.right_rgb,
            "left_white_rgb": self.left_white_rgb,
            "right_white_rgb": self.right_white_rgb
        }