import cv2 as cv
import numpy as np
from datasets.cubepp.cubepp_data import CubePPData
from datasets.cubepp.cubepp_dataloader import CubePPDataLoader
from helper import *
import sys

image_index = 0
if len(sys.argv) == 2:
    image_index = int(sys.argv[1])

cubepp_data_loader = CubePPDataLoader()
test_image = cubepp_data_loader.get(image_index)

print("Image Info:", test_image.get_info())

display_image = prepare_display(test_image.get_image(), correct_gamma=True)
cv.imshow("Test Image", display_image)
cv.waitKey(0)
cv.destroyAllWindows()