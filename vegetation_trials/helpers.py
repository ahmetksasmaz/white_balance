import numpy as np
import cv2 as cv

COLOR_POWER = 1/2.2  # Gamma correction for better visibility
# G = 1.0 and R = 1/255 -> log_u = log(1) - log(1/255) = log(255)
# G = 1/255 and R = 1.0 -> log_u = log(1/255) - log(1) = -log(255)
CHROMA_RANGE = np.log(255.0)  # Range of log chroma values to visualize
HIST_RANGE = [-0.1, 0.1] 

# Convert to log-chrominance space
def log_chrominance(image):
    b, g, r = cv.split(image)
    log_u = np.log1p(g) - np.log1p(r + 1e-8)
    log_v = np.log1p(g) - np.log1p(b + 1e-8)
    return log_u, log_v

def visualize_log_chroma():
    global CHROMA_RANGE
    g_channel = np.zeros((512, 512), dtype=np.float32)
    r_channel = np.zeros((512, 512), dtype=np.float32)
    b_channel = np.zeros((512, 512), dtype=np.float32)

    for i in range(512):
        for j in range(512):
            log_u = (i / 511.0) * CHROMA_RANGE * 2 - CHROMA_RANGE 
            log_v = (j / 511.0) * CHROMA_RANGE * 2 - CHROMA_RANGE
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

COLORMAP_BGR = visualize_log_chroma()


def draw_log_chroma_from_image(image):
    global COLORMAP_BGR, COLOR_POWER, HIST_RANGE
    input_log_u, input_log_v = log_chrominance(image)
    
    hist, xedges, yedges = np.histogram2d(input_log_u.flatten(), input_log_v.flatten(), bins=[512,512], range=[HIST_RANGE, HIST_RANGE])
    # Remove maximum %1 percentile to avoid outliers
    max_threshold = np.percentile(hist, 99)
    hist = np.clip(hist, 0, max_threshold)
    normalized_hist = hist / np.max(hist)
    # Apply colormap to histogram based on intensity
    hist_colored = np.zeros((512, 512, 3), dtype=np.float32)
    for i in range(512):
        for j in range(512):
            intensity = normalized_hist[i, j]
            hist_colored[i, j] = COLORMAP_BGR[i, j] * intensity
    hist_colored = np.power(hist_colored, COLOR_POWER)  # Gamma correction for better visibility
    hist_colored = hist_colored * 255.0
    hist_colored = np.clip(hist_colored, 0, 255)
    hist_colored = hist_colored.astype(np.uint8)
    # Draw axis at (0,0)
    center_x = int((0 - HIST_RANGE[0]) / (HIST_RANGE[1] - HIST_RANGE[0]) * 512)
    center_y = int((0 - HIST_RANGE[0]) / (HIST_RANGE[1] - HIST_RANGE[0]) * 512)
    cv.line(hist_colored, (center_x, 0), (center_x, 512-1), (255, 255, 255), 1)
    cv.line(hist_colored, (0, center_y), (512-1, center_y), (255, 255, 255), 1)

    return hist_colored, hist

def draw_log_chroma_from_histogram(hist):
    normalized_hist = hist / np.max(hist)
    # Apply colormap to histogram based on intensity
    hist_colored = np.zeros((512, 512, 3), dtype=np.float32)
    for i in range(512):
        for j in range(512):
            intensity = normalized_hist[i, j]
            hist_colored[i, j] = COLORMAP_BGR[i, j] * intensity
    hist_colored = np.power(hist_colored, COLOR_POWER)  # Gamma correction for better visibility
    hist_colored = hist_colored * 255.0
    hist_colored = np.clip(hist_colored, 0, 255)
    hist_colored = hist_colored.astype(np.uint8)
    # Draw axis at (0,0)
    center_x = int((0 - HIST_RANGE[0]) / (HIST_RANGE[1] - HIST_RANGE[0]) * 512)
    center_y = int((0 - HIST_RANGE[0]) / (HIST_RANGE[1] - HIST_RANGE[0]) * 512)
    cv.line(hist_colored, (center_x, 0), (center_x, 512-1), (255, 255, 255), 1)
    cv.line(hist_colored, (0, center_y), (512-1, center_y), (255, 255, 255), 1)

    return hist_colored

def display_image_with_wait(image, label, convert=True):
    if convert:
        display_image = image * 255.0
        display_image = np.clip(display_image, 0, 255).astype(np.uint8)
        display_image = cv.resize(display_image, (512, 512), interpolation=cv.INTER_LINEAR)
        cv.imshow(label, display_image)
        cv.waitKey(0)
    else:
        cv.imshow(label, image)
        cv.waitKey(0)