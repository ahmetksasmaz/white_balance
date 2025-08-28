import cv2 as cv
import numpy as np
from datasets.cubepp.cubepp_data import CubePPData
from datasets.cubepp.cubepp_dataloader import CubePPDataLoader
from helper import *
from errors import *
from chromatic_adaptation import *
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

def cubepp_process(data: CubePPData, algorithm: WhiteBalanceAlgorithm):
    orig_image = data.get_image()
    pred_illuminant = algorithm.estimate_global_illuminant(orig_image)
    target_illuminant = D65_WHITE_POINT
    target_illuminant.adjust_luminance(pred_illuminant.luminance())

    adapted_image = chromatic_adaptation(orig_image, pred_illuminant, target_illuminant)

    gt_image = None
    errors = {}

    if data.get_info()["mean_rgb"][0] is not None:
        mean_rgb = np.array(data.get_info()["mean_rgb"][::-1]) # RGB to BGR order
        gray_world_norm = np.linalg.norm(pred_illuminant.srgb())
        normalized_gt = mean_rgb / np.linalg.norm(mean_rgb)
        rescaled_gt = normalized_gt * gray_world_norm
        ground_truth_illuminant = WhitePoint(rescaled_gt, "SRGB")
        gt_target_illuminant = D65_WHITE_POINT
        gt_target_illuminant.adjust_luminance(ground_truth_illuminant.luminance())

        gt_image = chromatic_adaptation(orig_image, ground_truth_illuminant, gt_target_illuminant)

        angle_error = recovery_angular_error(ground_truth_illuminant.srgb(), pred_illuminant.srgb())
        square_error = recovery_square_error(ground_truth_illuminant.srgb(), pred_illuminant.srgb())
        ciede2000_error = recovery_ciede2000(ground_truth_illuminant.srgb(), pred_illuminant.srgb())

        errors = {
            "angle_error": angle_error,
            "square_error": square_error,
            "ciede2000_error": ciede2000_error
        }

    return orig_image, adapted_image, gt_image, errors