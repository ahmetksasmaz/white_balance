import cv2 as cv
import numpy as np
from datasets.cubepp.cubepp_data import CubePPData
from datasets.cubepp.cubepp_dataloader import CubePPDataLoader
from helper import *
from errors import *
from chromatic_adaptation import *
import sys

image_index = 0
if len(sys.argv) == 2:
    image_index = int(sys.argv[1])

cubepp_data_loader = CubePPDataLoader()
test_image = cubepp_data_loader.get(image_index)

if test_image.get_info()["mean_rgb"][0] is not None:
    mean_rgb = np.array(test_image.get_info()["mean_rgb"][::-1]) # RGB to BGR order
    image = test_image.get_image()

    gray_world_illuminant = np.mean(image, axis=(0, 1))
    print("Gray World Illuminant Estimate:", gray_world_illuminant)
    print("Ground Truth Illuminant:", mean_rgb)

    # Apply Chromatic Adaptation
    gray_world_illuminant = WhitePoint(gray_world_illuminant, "SRGB")
    target_illuminant = D65_WHITE_POINT
    target_illuminant.adjust_luminance(gray_world_illuminant.luminance())

    adapted_image = chromatic_adaptation(image, gray_world_illuminant, target_illuminant)

    # Apply Chromatic Adaptation with Ground Truth
    gray_world_norm = np.linalg.norm(gray_world_illuminant.srgb())
    normalized_gt = mean_rgb / np.linalg.norm(mean_rgb)
    mean_rgb *= gray_world_norm
    ground_truth_illuminant = WhitePoint(mean_rgb, "SRGB")
    gt_target_illuminant = D65_WHITE_POINT
    gt_target_illuminant.adjust_luminance(ground_truth_illuminant.luminance())

    gt_image = chromatic_adaptation(image, ground_truth_illuminant, gt_target_illuminant)

    adapted_display_image = prepare_display(adapted_image, correct_gamma=True)
    gt_display_image = prepare_display(gt_image, correct_gamma=True)
    display_image = prepare_display(image, correct_gamma=True)
    cv.imshow("Adapted Image", adapted_display_image)
    cv.imshow("Ground Truth Image", gt_display_image)
    cv.imshow("Original Image", display_image)

    cv.waitKey(0)
    cv.destroyAllWindows()