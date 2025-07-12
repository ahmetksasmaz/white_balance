import cv2 as cv
import numpy as np

# Load single image as YUV image
# Note: The image is in PNG format with 16-bit depth, so we need to
# read it with cv.IMREAD_UNCHANGED to preserve the depth.
image = cv.imread('images/SimpleCube++/train/PNG/00_0002.png', cv.IMREAD_UNCHANGED)

# Image is 432,648,3, uint16
# 14-bit quantization
# Convert to float32
black_level = 2048
saturation_level = 2**14-1
image = image.astype(np.float32)
# Linearize the image
image = np.clip((image - black_level) / (saturation_level - black_level) , 0, 1)

# Apply gamma correction
gamma = 2.2
image = np.power(image, 1.0 / gamma)

def apply_white_balance(image):
    image_corrected = image.copy()

    target_color = np.mean(image, axis=(0, 1))
    print("Target color (mean): {}".format(target_color))

    color_gain = 0.5 / target_color
    print("Color gain: {}".format(color_gain))

    # Apply color gain
    image_corrected = image * color_gain
    # Clip values to [0, 1]
    image_corrected = np.clip(image_corrected, 0, 1)
    # Convert back to uint8
    image_corrected = (image_corrected * 255).astype(np.uint8)

    return image_corrected

cv.imshow('image', apply_white_balance(image))
cv.waitKey(0)
cv.destroyAllWindows()