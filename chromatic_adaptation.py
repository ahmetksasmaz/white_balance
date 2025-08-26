import cv2 as cv
import numpy as np
from helper import *
np.seterr(all='ignore')

class CAT:
    def __init__(self, adaptation_matrix_name = "bradford"):
        self.adaptation_matrix_name = adaptation_matrix_name.lower()

        if self.adaptation_matrix_name == "bradford":
            self.M = np.array([[0.8951, 0.2664, -0.1614],[-0.7502, 1.7135, 0.0367],[0.0389, -0.0685, 1.0296]])
        elif self.adaptation_matrix_name == "von_kries":
            self.M = np.array([[0.40024, 0.70760, -0.08081],[-0.22630, 1.16532, 0.04570],[0.00000, 0.00000, 0.91822]])
        elif self.adaptation_matrix_name == "sharp":
            self.M = np.array([[1.2694, -0.0988, -0.1706],[-0.8364, 1.8006, 0.0357],[0.0297, -0.0315, 1.0018]])
        elif self.adaptation_matrix_name == "cat2000":
            self.M = np.array([[0.7982, 0.3389, -0.1371],[-0.5918, 1.5512, 0.0406],[0.0008, 0.2390, 0.9753]])
        elif self.adaptation_matrix_name == "cat02":
            self.M = np.array([[0.7328, 0.4296, -0.1624],[-0.7036, 1.6975, 0.0061],[0.0030, 0.0136, 0.9834]])

class WhitePoint:
    def __init__(self, illuminant, color_space = "SRGB"):
        color_space = color_space.upper()
        if color_space == "SRGB":
            self.srgb_color = illuminant
            self.xyz_color = srgb_to_xyz(illuminant)
            self.lab_color = srgb_to_lab(illuminant)
        elif color_space == "XYZ":
            self.xyz_color = illuminant
            self.srgb_color = xyz_to_srgb(illuminant)
            self.lab_color = xyz_to_lab(illuminant)
        elif color_space == "LAB":
            self.srgb_color = lab_to_srgb(illuminant)
            self.lab_color = illuminant
            self.xyz_color = lab_to_xyz(illuminant)

    def adjust_luminance(self, luminance):
        self.lab_color[0] = luminance
        self.xyz_color = lab_to_xyz(self.lab_color)
        self.srgb_color = lab_to_srgb(self.lab_color)

    def luminance(self):
        return self.lab_color[0]

    def srgb(self):
        return self.srgb_color

    def xyz(self):
        return self.xyz_color

    def lab(self):
        return self.lab_color

A_WHITE_POINT = WhitePoint(np.array([1.09850,1.00000,0.35585]), "xyz")
B_WHITE_POINT = WhitePoint(np.array([0.99072,1.00000,0.85223]), "xyz")
C_WHITE_POINT = WhitePoint(np.array([0.98074,1.00000,1.18232]), "xyz")
D50_WHITE_POINT = WhitePoint(np.array([0.96422,1.00000,0.82521]), "xyz")
D55_WHITE_POINT = WhitePoint(np.array([0.95682,1.00000,0.92149]), "xyz")
D65_WHITE_POINT = WhitePoint(np.array([0.95047,1.00000,1.08883]), "xyz")
D75_WHITE_POINT = WhitePoint(np.array([0.94972,1.00000,1.22638]), "xyz")
E_WHITE_POINT = WhitePoint(np.array([1.00000,1.00000,1.00000]), "xyz")
F2_WHITE_POINT = WhitePoint(np.array([0.99186,1.00000,0.67393]), "xyz")
F7_WHITE_POINT = WhitePoint(np.array([0.95041,1.00000,1.08747]), "xyz")
F11_WHITE_POINT = WhitePoint(np.array([1.00962,1.00000,0.64350]), "xyz")

BRADFORD_CAT = CAT("bradford")
VON_KRIES_CAT = CAT("von_kries")
SHARP_CAT = CAT("sharp")
CAT2000_CAT = CAT("cat2000")
CAT02_CAT = CAT("cat02")

def chromatic_adaptation(image:cv.Mat, source_illumination:WhitePoint, target_illumination:WhitePoint = D65_WHITE_POINT, cat:CAT = BRADFORD_CAT):
    adapted_image = image.copy()

    source_xyz = source_illumination.xyz()
    target_xyz = target_illumination.xyz()

    source_lms = cat.M @ np.array(source_xyz)
    target_lms = cat.M @ np.array(target_xyz)

    gain = target_lms / source_lms

    M_srgb_to_xyz = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])

    flat_adapted_image = adapted_image.reshape(-1, 3)
    flat_adapted_image = (M_srgb_to_xyz @ flat_adapted_image.T).T
    flat_adapted_image = (cat.M @ flat_adapted_image.T).T
    flat_adapted_image = flat_adapted_image * gain
    flat_adapted_image = (np.linalg.inv(cat.M) @ flat_adapted_image.T).T
    flat_adapted_image = (np.linalg.inv(M_srgb_to_xyz) @ flat_adapted_image.T).T

    adapted_image = flat_adapted_image.reshape(adapted_image.shape)
    adapted_image = np.clip(adapted_image, 0, 1)

    return adapted_image

# def chromatic_adaptation_map(image:cv.Mat, source_illumination_map: list[list[WhitePoint]], target_illumination:WhitePoint = D65_WHITE_POINT, cat:CAT = BRADFORD_CAT):
#     adapted_image = image.copy()
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             source_illumination = source_illumination_map[i][j]
#             pixel = image[i, j]
#             adapted_pixel = adapt_single_pixel(pixel, source_illumination, target_illumination, cat)
#             adapted_image[i, j] = adapted_pixel
#     return adapted_image