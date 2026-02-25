import cv2 as cv
import numpy as np
from datasets.data import Data
from datasets.cubepp.cubepp_dataprovider import CubePPDataProvider
from helper import *
import sys

image_index = 0
if len(sys.argv) == 2:
    image_index = int(sys.argv[1])

cubepp_data_provider = CubePPDataProvider()
test_data = cubepp_data_provider[image_index]

print("Image Name:", test_data.get_image_name())
print("Image Dimensions:", test_data.get_image_dimensions())
print("GT Info:", test_data.get_illuminants())