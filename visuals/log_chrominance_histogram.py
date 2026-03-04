import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from datasets.data import Data
from visuals.visuals import Visuals
from helper import log_chrominance, histogram, pdf, prepare_display

class LogChrominanceHistogram(Visuals):
    def __init__(self, image_size=512):
        super().__init__(image_size)

        self.COLOR_POWER = 1/2.8  # Gamma correction for better visibility
        # G = 1.0 and R = 1/255 -> log_u = log(1) - log(1/255) = log(255)
        # G = 1/255 and R = 1.0 -> log_u = log(1/255) - log(1) = -log(255)
        self.CHROMA_RANGE = np.log(255.0)  # Range of log chroma values to visualize
        self.HISTOGRAM_MULTIPLIER = 0.25 # To zoom in the histogram visualization
        self.HIST_RANGE = [-self.CHROMA_RANGE*self.HISTOGRAM_MULTIPLIER, self.CHROMA_RANGE*self.HISTOGRAM_MULTIPLIER] 
        self.CORNER_LINE_LENGTH = 25
        self.CORNER_LINE_THICKNESS = 2
        self.LOG_COLORMAP_BGR = self.__visualize_log_chroma()
    
    def _visualize(self, data):
        return self.__draw_log_chroma_from_image(data)

    def __draw_log_chroma_from_image(self, data):
        image = data.get_raw_image()
        input_log_u, input_log_v = self.__log_chrominance(image)
        
        # Use image_size for bins to match colormap
        bins = self.image_size
        hist, xedges, yedges = np.histogram2d(input_log_u.flatten(), input_log_v.flatten(), bins=[bins, bins], range=[self.HIST_RANGE, self.HIST_RANGE])
        
        normalized_hist = hist / (np.max(hist) + 1e-8)
        
        # Apply colormap to histogram based on intensity
        hist_colored = np.zeros((bins, bins, 3), dtype=np.float32)
        for i in range(bins):
            for j in range(bins):
                intensity = normalized_hist[i, j]
                hist_colored[i, j] = self.LOG_COLORMAP_BGR[i, j] * intensity
        
        hist_colored = np.power(hist_colored, self.COLOR_POWER)  # Gamma correction for better visibility
        # Normalize instead of clipping
        max_val = np.max(hist_colored)
        if max_val > 0:
            hist_colored = hist_colored / max_val
        hist_colored = (hist_colored * 255.0).astype(np.uint8)
        # Draw axis at (0,0)
        # center_x = int((0 - HIST_RANGE[0]) / (HIST_RANGE[1] - HIST_RANGE[0]) * quantization)
        # center_y = int((0 - HIST_RANGE[0]) / (HIST_RANGE[1] - HIST_RANGE[0]) * quantization)
        # cv.line(hist_colored, (center_x, 0), (center_x, quantization-1), (255, 255, 255), 1)
        # cv.line(hist_colored, (0, center_y), (quantization-1, center_y), (255, 255, 255), 1)

        # Draw lines to corners of the image for reference
        cv.line(hist_colored, (0, 0), (self.CORNER_LINE_LENGTH, self.CORNER_LINE_LENGTH), (self.LOG_COLORMAP_BGR[0, 0]*255).astype(np.uint8).tolist(), self.CORNER_LINE_THICKNESS)
        cv.line(hist_colored, (bins - 1, 0), (bins - 1 - self.CORNER_LINE_LENGTH, self.CORNER_LINE_LENGTH), (self.LOG_COLORMAP_BGR[0, bins - 1]*255).astype(np.uint8).tolist(), self.CORNER_LINE_THICKNESS)
        cv.line(hist_colored, (0, bins - 1), (self.CORNER_LINE_LENGTH, bins - 1 - self.CORNER_LINE_LENGTH), (self.LOG_COLORMAP_BGR[bins - 1, 0]*255).astype(np.uint8).tolist(), self.CORNER_LINE_THICKNESS)
        cv.line(hist_colored, (bins - 1, bins - 1), (bins - 1 - self.CORNER_LINE_LENGTH, bins - 1 - self.CORNER_LINE_LENGTH), (self.LOG_COLORMAP_BGR[bins - 1, bins - 1]*255).astype(np.uint8).tolist(), self.CORNER_LINE_THICKNESS)
 
        # Draw histogram multiplier text
        cv.putText(hist_colored, f"Zoom: {(1/self.HISTOGRAM_MULTIPLIER):0.2f}x", (5, bins//2 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv.LINE_AA)

        return hist_colored, hist

    def __log_chrominance(self, raw_image):
        mask = (raw_image[..., 0] != 0) | (raw_image[..., 1] != 0) | (raw_image[..., 2] != 0)
        masked_image = raw_image[mask]
        masked_image = np.reshape(masked_image, (-1, 1, 3))
        b, g, r = cv.split(masked_image)
        log_u = np.log(g + 1e-8) - np.log(r + 1e-8)
        log_v = np.log(g + 1e-8) - np.log(b + 1e-8)
        return log_u, log_v
    
    def __visualize_log_chroma(self):
        quantization = self.image_size  # Assuming square image for histogram
        g_channel = np.zeros((quantization, quantization), dtype=np.float32)
        r_channel = np.zeros((quantization, quantization), dtype=np.float32)
        b_channel = np.zeros((quantization, quantization), dtype=np.float32)

        for i in range(quantization):
            for j in range(quantization):
                log_u = (i / (quantization - 1)) * self.CHROMA_RANGE * 2 * self.HISTOGRAM_MULTIPLIER - self.CHROMA_RANGE * self.HISTOGRAM_MULTIPLIER
                log_v = (j / (quantization - 1)) * self.CHROMA_RANGE * 2 * self.HISTOGRAM_MULTIPLIER - self.CHROMA_RANGE * self.HISTOGRAM_MULTIPLIER
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