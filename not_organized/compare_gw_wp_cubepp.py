import cv2 as cv
import numpy as np

# Note: The image is in PNG format with 16-bit depth, so we need to
# read it with cv.IMREAD_UNCHANGED to preserve the depth.
image = cv.imread('00_0002.png', cv.IMREAD_UNCHANGED)

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

def apply_gw(image):
    image_corrected = image.copy()

    target_color = np.mean(image, axis=(0, 1))
    print("Target color (mean): {}".format(target_color))

    color_gain = target_color[1] / target_color # base to green channel
    print("Color gain: {}".format(color_gain))

    # Apply color gain
    image_corrected = image * color_gain
    # Clip values to [0, 1]
    image_corrected = np.clip(image_corrected, 0, 1)
    # Convert back to uint8
    image_corrected = (image_corrected * 255).astype(np.uint8)

    cv.putText(image_corrected, "Gray World", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image_corrected

def apply_wp(image):
    image_corrected = image.copy()

    target_color = np.max(image, axis=(0, 1))
    print("Target color (mean): {}".format(target_color))

    color_gain = target_color[1] / target_color # base to green channel "gmax / rmax"
    print("Color gain: {}".format(color_gain))

    # Apply color gain
    image_corrected = image * color_gain
    # Clip values to [0, 1]
    image_corrected = np.clip(image_corrected, 0, 1)
    # Convert back to uint8
    image_corrected = (image_corrected * 255).astype(np.uint8)

    cv.putText(image_corrected, "White Patch", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image_corrected

display_image = np.zeros((image.shape[0], image.shape[1] * 2, 3), dtype=np.uint8)
display_image[:, :image.shape[1], :] = apply_gw(image)
display_image[:, image.shape[1]:, :] = apply_wp(image)

cv.imshow('image', display_image)
cv.waitKey(0)
cv.destroyAllWindows()