import numpy as np
import cv2 as cv

COLOR_POWER = 1/2.8  # Gamma correction for better visibility
# G = 1.0 and R = 1/255 -> log_u = log(1) - log(1/255) = log(255)
# G = 1/255 and R = 1.0 -> log_u = log(1/255) - log(1) = -log(255)
CHROMA_RANGE = np.log(255.0)  # Range of log chroma values to visualize
HISTOGRAM_MULTIPLIER = 0.25 # To zoom in the histogram visualization
HIST_RANGE = [-CHROMA_RANGE*HISTOGRAM_MULTIPLIER, CHROMA_RANGE*HISTOGRAM_MULTIPLIER] 
RGB_HIST_RANGE = [0, 1.0]
CORNER_LINE_LENGTH = 25
CORNER_LINE_THICKNESS = 2
HISTOGRAM_BINS = 256

# Convert to log-chrominance space
def log_chrominance(image):
    mask = (image[..., 0] != 0) | (image[..., 1] != 0) | (image[..., 2] != 0)
    masked_image = image[mask]
    masked_image = np.reshape(masked_image, (-1, 1, 3))
    b, g, r = cv.split(masked_image)
    log_u = np.log(g + 1e-8) - np.log(r + 1e-8)
    log_v = np.log(g + 1e-8) - np.log(b + 1e-8)
    return log_u, log_v

def rgb_chrominance(image):
    # Skip pixels with 0,0,0 value
    mask = (image[..., 0] != 0) | (image[..., 1] != 0) | (image[..., 2] != 0)
    masked_image = image[mask]
    masked_image = np.reshape(masked_image, (-1, 1, 3))
    b, g, r = cv.split(masked_image)
    sum_channels = r + g + b + 1e-8
    r_c = r / sum_channels
    g_c = g / sum_channels
    b_c = b / sum_channels
    return b_c, g_c, r_c

def visualize_rgb_chroma():
    g_channel = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS), dtype=np.float32)
    r_channel = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS), dtype=np.float32)
    b_channel = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS), dtype=np.float32)

    for i in range(HISTOGRAM_BINS):
        for j in range(HISTOGRAM_BINS):
            r_c = (i / (HISTOGRAM_BINS - 1))
            b_c = (j / (HISTOGRAM_BINS - 1))
            if r_c + b_c > 1.0:
                continue
            g_c = 1.0 - r_c - b_c
            sum_c = r_c + g_c + b_c + 1e-8
            r = r_c / sum_c
            g = g_c / sum_c
            b = b_c / sum_c
            r_channel[i, j] = np.clip(r, 0, 1)
            g_channel[i, j] = np.clip(g, 0, 1)
            b_channel[i, j] = np.clip(b, 0, 1)
    colormap_bgr = cv.merge((b_channel, g_channel, r_channel))
    return colormap_bgr

def visualize_log_chroma():
    global CHROMA_RANGE
    g_channel = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS), dtype=np.float32)
    r_channel = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS), dtype=np.float32)
    b_channel = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS), dtype=np.float32)

    for i in range(HISTOGRAM_BINS):
        for j in range(HISTOGRAM_BINS):
            log_u = (i / (HISTOGRAM_BINS - 1)) * CHROMA_RANGE * 2 * HISTOGRAM_MULTIPLIER - CHROMA_RANGE * HISTOGRAM_MULTIPLIER
            log_v = (j / (HISTOGRAM_BINS - 1)) * CHROMA_RANGE * 2 * HISTOGRAM_MULTIPLIER - CHROMA_RANGE * HISTOGRAM_MULTIPLIER
            l_u = np.exp(-log_u)
            l_v = np.exp(-log_v)
            z = np.sqrt(l_u**2 + l_v**2 + 1.0)
            r = l_u / z
            g = 1.0 / z
            b = l_v / z
            r_channel[i, j] = np.clip(r, 0, 1)
            g_channel[i, j] = np.clip(g, 0, 1)
            b_channel[i, j] = np.clip(b, 0, 1)
    colormap_bgr = cv.merge((b_channel, g_channel, r_channel))
    return colormap_bgr

LOG_COLORMAP_BGR = visualize_log_chroma()
RGB_COLORMAP_BGR = visualize_rgb_chroma()

def draw_rgb_chroma_from_image(image):
    global RGB_COLORMAP_BGR, COLOR_POWER, RGB_HIST_RANGE
    input_b_c, input_g_c, input_r_c = rgb_chrominance(image)
    
    hist, xedges, yedges = np.histogram2d(input_r_c.flatten(), input_b_c.flatten(), bins=[HISTOGRAM_BINS,HISTOGRAM_BINS], range=[RGB_HIST_RANGE, RGB_HIST_RANGE])
    normalized_hist = hist / np.max(hist)
    # Apply colormap to histogram based on intensity
    hist_colored = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS, 3), dtype=np.float32)
    for i in range(HISTOGRAM_BINS):
        for j in range(HISTOGRAM_BINS):
            intensity = normalized_hist[i, j]
            hist_colored[i, j] = RGB_COLORMAP_BGR[i, j] * intensity
    hist_colored = np.power(hist_colored, COLOR_POWER)  # Gamma correction for better visibility
    hist_colored = hist_colored * 255.0
    hist_colored = np.clip(hist_colored, 0, 255)
    hist_colored = hist_colored.astype(np.uint8)
    cv.line(hist_colored, (0, HISTOGRAM_BINS - 1), (HISTOGRAM_BINS - 1, 0), (255, 255, 255), 3)

    # Draw lines to corners of the image for reference
    center_x = (0 + 0 + HISTOGRAM_BINS - 1) // 3
    center_y = (0 + 0 + HISTOGRAM_BINS - 1) // 3
    line_1_vector = (0 - center_x, 0 - center_y)
    line_2_vector = (HISTOGRAM_BINS - 1 - center_x, 0 - center_y)
    line_3_vector = (0 - center_x, HISTOGRAM_BINS - 1 - center_y)
    line_1_normalized = (line_1_vector[0] / np.sqrt(line_1_vector[0]**2 + line_1_vector[1]**2), line_1_vector[1] / np.sqrt(line_1_vector[0]**2 + line_1_vector[1]**2))
    line_2_normalized = (line_2_vector[0] / np.sqrt(line_2_vector[0]**2 + line_2_vector[1]**2), line_2_vector[1] / np.sqrt(line_2_vector[0]**2 + line_2_vector[1]**2))
    line_3_normalized = (line_3_vector[0] / np.sqrt(line_3_vector[0]**2 + line_3_vector[1]**2), line_3_vector[1] / np.sqrt(line_3_vector[0]**2 + line_3_vector[1]**2))
    cv.line(hist_colored, (0, 0), (int(0 - line_1_normalized[0] * CORNER_LINE_LENGTH), int(0 - line_1_normalized[1] * CORNER_LINE_LENGTH)), (RGB_COLORMAP_BGR[0, 0]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)
    cv.line(hist_colored, (HISTOGRAM_BINS - 1, 0), (HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH, int(0 - line_2_normalized[1] * CORNER_LINE_LENGTH)), (RGB_COLORMAP_BGR[0, HISTOGRAM_BINS - 1]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)
    cv.line(hist_colored, (0, HISTOGRAM_BINS - 1), (int(0 - line_3_normalized[0] * CORNER_LINE_LENGTH), HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH), (RGB_COLORMAP_BGR[HISTOGRAM_BINS - 1, 0]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)

    return hist_colored, hist

def draw_rgb_chroma_from_histogram(hist):
    normalized_hist = hist / np.max(hist)
    # Apply colormap to histogram based on intensity
    hist_colored = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS, 3), dtype=np.float32)
    for i in range(HISTOGRAM_BINS):
        for j in range(HISTOGRAM_BINS):
            intensity = normalized_hist[i, j]
            hist_colored[i, j] = RGB_COLORMAP_BGR[i, j] * intensity
    hist_colored = np.power(hist_colored, COLOR_POWER)  # Gamma correction for better visibility
    hist_colored = hist_colored * 255.0
    hist_colored = np.clip(hist_colored, 0, 255)
    hist_colored = hist_colored.astype(np.uint8)
    cv.line(hist_colored, (0, HISTOGRAM_BINS - 1), (HISTOGRAM_BINS - 1, 0), (255, 255, 255), 3)

    # Draw lines to corners of the image for reference
    center_x = (0 + 0 + HISTOGRAM_BINS - 1) // 3
    center_y = (0 + 0 + HISTOGRAM_BINS - 1) // 3
    line_1_vector = (0 - center_x, 0 - center_y)
    line_2_vector = (HISTOGRAM_BINS - 1 - center_x, 0 - center_y)
    line_3_vector = (0 - center_x, HISTOGRAM_BINS - 1 - center_y)
    line_1_normalized = (line_1_vector[0] / np.sqrt(line_1_vector[0]**2 + line_1_vector[1]**2), line_1_vector[1] / np.sqrt(line_1_vector[0]**2 + line_1_vector[1]**2))
    line_2_normalized = (line_2_vector[0] / np.sqrt(line_2_vector[0]**2 + line_2_vector[1]**2), line_2_vector[1] / np.sqrt(line_2_vector[0]**2 + line_2_vector[1]**2))
    line_3_normalized = (line_3_vector[0] / np.sqrt(line_3_vector[0]**2 + line_3_vector[1]**2), line_3_vector[1] / np.sqrt(line_3_vector[0]**2 + line_3_vector[1]**2))
    cv.line(hist_colored, (0, 0), (int(0 - line_1_normalized[0] * CORNER_LINE_LENGTH), int(0 - line_1_normalized[1] * CORNER_LINE_LENGTH)), (RGB_COLORMAP_BGR[0, 0]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)
    cv.line(hist_colored, (HISTOGRAM_BINS - 1, 0), (HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH, int(0 - line_2_normalized[1] * CORNER_LINE_LENGTH)), (RGB_COLORMAP_BGR[0, HISTOGRAM_BINS - 1]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)
    cv.line(hist_colored, (0, HISTOGRAM_BINS - 1), (int(0 - line_3_normalized[0] * CORNER_LINE_LENGTH), HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH), (RGB_COLORMAP_BGR[HISTOGRAM_BINS - 1, 0]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)

    return hist_colored

def draw_log_chroma_from_image(image):
    global LOG_COLORMAP_BGR, COLOR_POWER, HIST_RANGE
    input_log_u, input_log_v = log_chrominance(image)
    
    hist, xedges, yedges = np.histogram2d(input_log_u.flatten(), input_log_v.flatten(), bins=[HISTOGRAM_BINS,HISTOGRAM_BINS], range=[HIST_RANGE, HIST_RANGE])
    normalized_hist = hist / np.max(hist)
    # Apply colormap to histogram based on intensity
    hist_colored = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS, 3), dtype=np.float32)
    for i in range(HISTOGRAM_BINS):
        for j in range(HISTOGRAM_BINS):
            intensity = normalized_hist[i, j]
            hist_colored[i, j] = LOG_COLORMAP_BGR[i, j] * intensity
    hist_colored = np.power(hist_colored, COLOR_POWER)  # Gamma correction for better visibility
    hist_colored = hist_colored * 255.0
    hist_colored = np.clip(hist_colored, 0, 255)
    hist_colored = hist_colored.astype(np.uint8)
    # Draw axis at (0,0)
    # center_x = int((0 - HIST_RANGE[0]) / (HIST_RANGE[1] - HIST_RANGE[0]) * HISTOGRAM_BINS)
    # center_y = int((0 - HIST_RANGE[0]) / (HIST_RANGE[1] - HIST_RANGE[0]) * HISTOGRAM_BINS)
    # cv.line(hist_colored, (center_x, 0), (center_x, HISTOGRAM_BINS-1), (255, 255, 255), 1)
    # cv.line(hist_colored, (0, center_y), (HISTOGRAM_BINS-1, center_y), (255, 255, 255), 1)

    # Draw lines to corners of the image for reference
    cv.line(hist_colored, (0, 0), (CORNER_LINE_LENGTH, CORNER_LINE_LENGTH), (LOG_COLORMAP_BGR[0, 0]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)
    cv.line(hist_colored, (HISTOGRAM_BINS - 1, 0), (HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH, CORNER_LINE_LENGTH), (LOG_COLORMAP_BGR[0, HISTOGRAM_BINS - 1]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)
    cv.line(hist_colored, (0, HISTOGRAM_BINS - 1), (CORNER_LINE_LENGTH, HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH), (LOG_COLORMAP_BGR[HISTOGRAM_BINS - 1, 0]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)
    cv.line(hist_colored, (HISTOGRAM_BINS - 1, HISTOGRAM_BINS - 1), (HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH, HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH), (LOG_COLORMAP_BGR[HISTOGRAM_BINS - 1, HISTOGRAM_BINS - 1]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)

    # Draw histogram multiplier text
    cv.putText(hist_colored, f"Zoom: {(1/HISTOGRAM_MULTIPLIER):0.2f}x", (5, HISTOGRAM_BINS//2 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv.LINE_AA)

    return hist_colored, hist

def draw_log_chroma_from_histogram(hist):
    normalized_hist = hist / np.max(hist)
    # Apply colormap to histogram based on intensity
    hist_colored = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS, 3), dtype=np.float32)
    for i in range(HISTOGRAM_BINS):
        for j in range(HISTOGRAM_BINS):
            intensity = normalized_hist[i, j]
            hist_colored[i, j] = LOG_COLORMAP_BGR[i, j] * intensity
    hist_colored = np.power(hist_colored, COLOR_POWER)  # Gamma correction for better visibility
    hist_colored = hist_colored * 255.0
    hist_colored = np.clip(hist_colored, 0, 255)
    hist_colored = hist_colored.astype(np.uint8)
    # Draw axis at (0,0)
    # center_x = int((0 - HIST_RANGE[0]) / (HIST_RANGE[1] - HIST_RANGE[0]) * HISTOGRAM_BINS)
    # center_y = int((0 - HIST_RANGE[0]) / (HIST_RANGE[1] - HIST_RANGE[0]) * HISTOGRAM_BINS)
    # cv.line(hist_colored, (center_x, 0), (center_x, HISTOGRAM_BINS-1), (255, 255, 255), 1)
    # cv.line(hist_colored, (0, center_y), (HISTOGRAM_BINS-1, center_y), (255, 255, 255), 1)

    # Draw lines to corners of the image for reference
    cv.line(hist_colored, (0, 0), (CORNER_LINE_LENGTH, CORNER_LINE_LENGTH), (LOG_COLORMAP_BGR[0, 0]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)
    cv.line(hist_colored, (HISTOGRAM_BINS - 1, 0), (HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH, CORNER_LINE_LENGTH), (LOG_COLORMAP_BGR[0, HISTOGRAM_BINS - 1]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)
    cv.line(hist_colored, (0, HISTOGRAM_BINS - 1), (CORNER_LINE_LENGTH, HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH), (LOG_COLORMAP_BGR[HISTOGRAM_BINS - 1, 0]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)
    cv.line(hist_colored, (HISTOGRAM_BINS - 1, HISTOGRAM_BINS - 1), (HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH, HISTOGRAM_BINS - 1 - CORNER_LINE_LENGTH), (LOG_COLORMAP_BGR[HISTOGRAM_BINS - 1, HISTOGRAM_BINS - 1]*255).astype(np.uint8).tolist(), CORNER_LINE_THICKNESS)

    cv.putText(hist_colored, f"Zoom: {(1/HISTOGRAM_MULTIPLIER):0.2f}x", (5, HISTOGRAM_BINS//2 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv.LINE_AA)

    return hist_colored

def display_image_with_wait(image, label, convert=True):
    if convert:
        display_image = image * 255.0
        display_image = np.clip(display_image, 0, 255).astype(np.uint8)
        display_image = cv.resize(display_image, (HISTOGRAM_BINS, HISTOGRAM_BINS), interpolation=cv.INTER_LINEAR)
        cv.imshow(label, display_image)
        cv.waitKey(0)
    else:
        cv.imshow(label, image)
        cv.waitKey(0)