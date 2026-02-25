import cv2 as cv
import numpy as np
from datasets.data import Data
from datasets.lsmi.lsmi_dataprovider import LSMIDataProvider
from helper import *
import sys

image_index = 0
if len(sys.argv) == 2:
    image_index = int(sys.argv[1])

lsmi_data_provider = LSMIDataProvider()
test_image = lsmi_data_provider[image_index]

test_data = lsmi_data_provider[image_index]

print("Image Name:", test_data.get_image_name())
print("Image Dimensions:", test_data.get_image_dimensions())
print("GT Info:", test_data.get_illuminants())
print("Illuminant Map", test_data.get_illuminant_map())