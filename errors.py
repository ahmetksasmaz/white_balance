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

def _worker_error_map(shm_name, shape, dtype, gt_image, adapted_image, start_row, end_row):
    shm = shared_memory.SharedMemory(name=shm_name)
    error_map = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    width = shape[1]
    for h in range(start_row, end_row):
        for w in range(width):
            error_map[h, w] = recovery_ciede2000(gt_image[h, w], adapted_image[h, w])
    shm.close()

def create_error_heatmap(adapted_image, gt_image):
    height, width, _ = adapted_image.shape
    error_map = np.zeros((height, width), dtype=np.float32)

    # Use multithreading to speed up the process
    # Create shared memory for error_map
    shm = shared_memory.SharedMemory(create=True, size=error_map.nbytes)
    shm_error_map = np.ndarray(error_map.shape, dtype=error_map.dtype, buffer=shm.buf)

    num_threads = os.cpu_count() or 4
    rows_per_thread = height // num_threads
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start_row = i * rows_per_thread
            end_row = (i + 1) * rows_per_thread if i != num_threads - 1 else height
            # Slices of gt_image and adapted_image are pickled, so pass full arrays
            futures.append(executor.submit(
                _worker_error_map,
                shm.name,
                error_map.shape,
                error_map.dtype,
                gt_image,
                adapted_image,
                start_row,
                end_row
            ))
        concurrent.futures.wait(futures)

    error_map = np.nan_to_num(shm_error_map, nan=0.0, posinf=0.0, neginf=0.0)
    shm.close()
    shm.unlink()
    # Multi threading end

    max_error = np.max(error_map)
    mean_error = np.mean(error_map)

    heatmap = cv.applyColorMap(((np.clip(error_map, 0.0, max_error) / max_error) * 255).astype(np.uint8), cv.COLORMAP_JET)
    # Draw colormap on top right corner
    colormap_height = height // 3
    colormap_width = width // 50
    colormap = np.linspace(max_error, 0, colormap_height).astype(np.float32)
    colormap_img = cv.applyColorMap(((colormap / max_error) * 255).astype(np.uint8), cv.COLORMAP_JET)
    colormap_img = colormap_img.reshape((colormap_height, 1, 3))
    colormap_img = np.repeat(colormap_img, colormap_width, axis=1)

    # Draw black line for mean_error ratio
    line_pos = int((1 - min(mean_error / max_error, 1.0)) * colormap_height)
    cv.line(colormap_img, (0, line_pos), (colormap_width - 1, line_pos), (0, 0, 0), 10)

    # Write max_error value with parenthesis
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 2
    text = f"({max_error:.2f})"
    text_size, _ = cv.getTextSize(text, font, font_scale, thickness)
    text_x = heatmap.shape[1] - colormap_width - text_size[0] - 5
    text_y = text_size[1] + 10
    # Draw square for text background
    cv.rectangle(heatmap, (heatmap.shape[1] - colormap_width - text_size[0] - 10, 0), (heatmap.shape[1] - colormap_width, text_size[1] + 30), (0, 0, 0), -1)
    cv.putText(heatmap, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

    # Write mean_error value with brackets
    text = f"[{mean_error:.2f}]"
    text_size, _ = cv.getTextSize(text, font, font_scale, thickness)
    text_x = heatmap.shape[1] - colormap_width - text_size[0] - 5
    text_y = line_pos + 20
    # Draw square for text background
    cv.rectangle(heatmap, (heatmap.shape[1] - colormap_width - text_size[0] - 10, line_pos - text_size[1] - 5), (heatmap.shape[1] - colormap_width, line_pos + text_size[1] + 5), (0, 0, 0), -1)
    cv.putText(heatmap, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

    # Overlay colormap on top right corner
    heatmap[:colormap_height, -colormap_width:] = colormap_img

    heatmap = heatmap.astype(np.float32) / 255.0
    return heatmap