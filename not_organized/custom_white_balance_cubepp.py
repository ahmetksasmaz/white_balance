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

def apply_white_balance(image, target_color_coordinate = None):
    image_corrected = image.copy()
    if target_color_coordinate is not None:
        if target_color_coordinate[0] < 0 or target_color_coordinate[0] > image.shape[1] - 1 or target_color_coordinate[1] < 0 or target_color_coordinate[1] > image.shape[0] - 1:
            image_corrected = np.clip(image_corrected, 0, 1)
            image_corrected = (image_corrected * 255).astype(np.uint8)
            return image_corrected
        target_color = image[target_color_coordinate[1], target_color_coordinate[0], :]
        print("Target color (at coordinate {}): {}".format(target_color_coordinate, target_color))
        color_gain = target_color[1] / target_color # base to green channel

        # Apply color gain
        image_corrected = image * color_gain
    # Clip values to [0, 1]
    image_corrected = np.clip(image_corrected, 0, 1)
    # Convert back to uint8
    image_corrected = (image_corrected * 255).astype(np.uint8)

    return image_corrected

def mouse_event(event, x, y, flags, param):
    global image, image_corrected, focus_roi, draw_image
    if event == cv.EVENT_LBUTTONDOWN:
        image_corrected = apply_white_balance(image, target_color_coordinate=(x, y))
        draw_image = prepare_draw_image(image_corrected, focus_roi)
    if event == cv.EVENT_MOUSEMOVE:
        focus_roi = prepare_focus_roi(image, target_color_coordinate=(x, y))
        draw_image = prepare_draw_image(image_corrected, focus_roi)

def prepare_focus_roi(image, target_color_coordinate=None):
    # Create a mask for the focus ROI
    focus_roi = np.zeros((180,180,3), dtype=np.uint8)
    if target_color_coordinate is None:
        target_color_coordinate = (image.shape[0] // 2, image.shape[1] // 2)
    if target_color_coordinate[0] < 1 or target_color_coordinate[0] > image.shape[1] - 2 or target_color_coordinate[1] < 1 or target_color_coordinate[1] > image.shape[0] - 2:
        return focus_roi
    for i in range(3):
        for j in range(3):
            focus_roi[i*60:(i+1)*60, j*60:(j+1)*60, :] = image[target_color_coordinate[1] - 1 + i, target_color_coordinate[0] - 1 + j, :] * 255
    return focus_roi


def prepare_draw_image(corrected_image, focus_roi):
    new_image = np.zeros((corrected_image.shape[0], corrected_image.shape[1] + focus_roi.shape[1], 3), dtype=np.uint8)
    new_image[:, :corrected_image.shape[1]] = corrected_image
    new_image[:focus_roi.shape[1], corrected_image.shape[1]:] = focus_roi
    return new_image


cv.namedWindow("image")
cv.setMouseCallback("image", mouse_event)

image_corrected = apply_white_balance(image)
focus_roi = prepare_focus_roi(image)

draw_image = prepare_draw_image(image_corrected, focus_roi)

while True:
    cv.imshow('image', draw_image)
    key = cv.waitKey(30) & 0xFF
    if key == 27:  # ESC key to exit
        break
    if key == ord('r'):
        image_corrected = apply_white_balance(image)
        focus_roi = prepare_focus_roi(image)
        draw_image = prepare_draw_image(image_corrected, focus_roi)

cv.destroyAllWindows()