import cv2 as cv
import numpy as np
import sys
from datasets.data import Data
from datasets.nus8.nus8_dataprovider import NUS8DataProvider
from helper import *

image_index = 0
if len(sys.argv) == 2:
    image_index = int(sys.argv[1])

nus8_data_provider = NUS8DataProvider()
test_data = nus8_data_provider[image_index]

print("Image Name:", test_data.get_image_name())
print("Image Dimensions:", test_data.get_image_dimensions())
print("GT Info:", test_data.get_illuminants())
