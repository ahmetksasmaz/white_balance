import numpy as np
import cv2 as cv
from helper import *
import concurrent.futures
import os
from multiprocessing import shared_memory

def recovery_angular_error(gt_illuminant, pred_illuminant):
    gt_illuminant = gt_illuminant / np.linalg.norm(gt_illuminant)
    pred_illuminant = pred_illuminant / np.linalg.norm(pred_illuminant)
    cos_theta = np.clip(np.dot(gt_illuminant, pred_illuminant), -1.0, 1.0)
    return np.arccos(cos_theta) * (180.0 / np.pi)

def recovery_square_error(gt_illuminant, pred_illuminant):
    gt_illuminant_quantized = np.round(gt_illuminant * 255)
    pred_illuminant_quantized = np.round(pred_illuminant * 255)
    return np.sum((gt_illuminant_quantized - pred_illuminant_quantized) ** 2)