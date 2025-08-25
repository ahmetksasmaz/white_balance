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

print("Image Info:", test_image.get_info())

if test_image.get_info()["mean_rgb"] is not None:
    mean_rgb = np.array(test_image.get_info()["mean_rgb"])
    print("Mean RGB:", mean_rgb)

    # Gray World Assumption
    gray_world_illuminant = np.mean(test_image.get_image(), axis=(0, 1))
    print("Gray World Illuminant Estimate:", gray_world_illuminant)

    # Apply Chromatic Adaptation
    adapted_image = chromatic_adaptation(test_image.get_image(), gray_world_illuminant, mean_rgb)

    adapted_display_image = prepare_display(adapted_image, correct_gamma=True)
    cv.imshow("Adapted Image", adapted_display_image)

    cv.waitKey(0)
    cv.destroyAllWindows()