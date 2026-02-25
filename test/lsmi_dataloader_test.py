import cv2 as cv
import numpy as np
from datasets.lsmi.lsmi_data import LSMIData
from datasets.lsmi.lsmi_dataloader import LSMIDataLoader
from helper import *
import sys

image_index = 0
if len(sys.argv) == 2:
    image_index = int(sys.argv[1])

lsmi_data_loader = LSMIDataLoader()
test_image = lsmi_data_loader[image_index]

print("Image path:", test_image.image_path)
print("Coeff path:", test_image.coeff_path)

print("Image Info:", test_image.get_info()["camera_model"])
print("Image Info:", test_image.get_info()["place"])
print("Image Info:", test_image.get_info()["lights"])

display_image = prepare_display(test_image.get_image(), correct_gamma=True)
cv.imshow("Test Image", display_image)
cv.waitKey(0)
display_gt = prepare_display(test_image.get_gt_image(), correct_gamma=True)
cv.imshow("GT Image", display_gt)
cv.waitKey(0)
cv.destroyAllWindows()