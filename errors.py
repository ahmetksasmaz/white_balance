import numpy as np
import cv2 as cv
from helper import *

def recovery_angular_error(gt_illuminant, pred_illuminant):
    gt_illuminant = gt_illuminant / np.linalg.norm(gt_illuminant)
    pred_illuminant = pred_illuminant / np.linalg.norm(pred_illuminant)
    cos_theta = np.clip(np.dot(gt_illuminant, pred_illuminant), -1.0, 1.0)
    return np.arccos(cos_theta) * (180.0 / np.pi)

def reproduction_angular_error(white_patch):
    return recovery_angular_error(np.array([1.0, 1.0, 1.0]), white_patch)

def recovery_square_error(gt_illuminant, pred_illuminant):
    gt_illuminant_quantized = np.round(gt_illuminant * 255)
    pred_illuminant_quantized = np.round(pred_illuminant * 255)
    return np.sum((gt_illuminant_quantized - pred_illuminant_quantized) ** 2)

def reproduction_square_error(white_patch):
    white_patch_quantized = np.round(white_patch * 255)
    return recovery_square_error(np.array([255, 255, 255]), white_patch_quantized)

def recovery_ciede2000(gt_illuminant, pred_illuminant):
    # Convert to LAB
    lab1 = srgb_to_lab(gt_illuminant)
    lab2 = srgb_to_lab(pred_illuminant)

    # CIEDE2000 implementation
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt((avg_C ** 7) / (avg_C ** 7 + 25 ** 7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)
    avg_Cp = (C1p + C2p) / 2.0

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    deltahp = 0
    if C1p * C2p == 0:
        deltahp = 0
    else:
        dh = h2p - h1p
        if abs(dh) <= 180:
            deltahp = dh
        elif dh > 180:
            deltahp = dh - 360
        else:
            deltahp = dh + 360

    deltaLp = L2 - L1
    deltaCp = C2p - C1p
    deltaHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(deltahp / 2))

    avg_Lp = (L1 + L2) / 2.0
    avg_Cp = (C1p + C2p) / 2.0

    hsum = h1p + h2p
    if C1p * C2p == 0:
        avg_hp = hsum
    else:
        dh = abs(h1p - h2p)
        if dh <= 180:
            avg_hp = hsum / 2.0
        else:
            if hsum < 360:
                avg_hp = (hsum + 360) / 2.0
            else:
                avg_hp = (hsum - 360) / 2.0

    T = 1 - 0.17 * np.cos(np.radians(avg_hp - 30)) + \
        0.24 * np.cos(np.radians(2 * avg_hp)) + \
        0.32 * np.cos(np.radians(3 * avg_hp + 6)) - \
        0.20 * np.cos(np.radians(4 * avg_hp - 63))

    delta_theta = 30 * np.exp(-((avg_hp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt((avg_Cp ** 7) / (avg_Cp ** 7 + 25 ** 7))
    Sl = 1 + ((0.015 * ((avg_Lp - 50) ** 2)) / np.sqrt(20 + ((avg_Lp - 50) ** 2)))
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -np.sin(np.radians(2 * delta_theta)) * Rc

    deltaE = np.sqrt(
        (deltaLp / Sl) ** 2 +
        (deltaCp / Sc) ** 2 +
        (deltaHp / Sh) ** 2 +
        Rt * (deltaCp / Sc) * (deltaHp / Sh)
    )
    return deltaE

def reproduction_ciede2000(white_patch):
    return recovery_ciede2000(np.array([1.0, 1.0, 1.0]), white_patch)