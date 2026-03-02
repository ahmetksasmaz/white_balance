import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from datasets.data import Data
from visuals.visuals import Visuals
from helper import log_chrominance, histogram, pdf, prepare_display

class NormalizedRGBHistogram(Visuals):
    def __init__(self, image_size=512):
        super().__init__(image_size)

        self.COLOR_POWER = 1/2.8  # Gamma correction for better visibility
        # G = 1.0 and R = 1/255 -> log_u = log(1) - log(1/255) = log(255)
        # G = 1/255 and R = 1.0 -> log_u = log(1/255) - log(1) = -log(255)
        self.HIST_RANGE = [0, 1.0]
        self.CORNER_LINE_LENGTH = 25
        self.CORNER_LINE_THICKNESS = 2
        self.RGB_COLORMAP_BGR = self.__visualize_rgb_chroma()
    
    def _visualize(self, data):
        return self.__draw_rgb_chroma_from_image(data)

    def __draw_rgb_chroma_from_image(self, data):
        image = data.get_raw_image()
        input_b_c, input_g_c, input_r_c = self.__rgb_chrominance(image)
        quantization = data.get_quantization()  # Assuming square image for histogram
        
        hist, xedges, yedges = np.histogram2d(input_r_c.flatten(), input_b_c.flatten(), bins=[quantization,quantization], range=[self.HIST_RANGE, self.HIST_RANGE])
        normalized_hist = hist / np.max(hist)
        # Apply colormap to histogram based on intensity
        hist_colored = np.zeros((quantization, quantization, 3), dtype=np.float32)
        for i in range(quantization):
            for j in range(quantization):
                intensity = normalized_hist[i, j]
                hist_colored[i, j] = self.RGB_COLORMAP_BGR[i, j] * intensity
        hist_colored = np.power(hist_colored, self.COLOR_POWER)  # Gamma correction for better visibility
        hist_colored = hist_colored * 255.0
        hist_colored = np.clip(hist_colored, 0, 255)
        hist_colored = hist_colored.astype(np.uint8)
        cv.line(hist_colored, (0, quantization - 1), (quantization - 1, 0), (255, 255, 255), 3)

        # Draw lines to corners of the image for reference
        center_x = (0 + 0 + quantization - 1) // 3
        center_y = (0 + 0 + quantization - 1) // 3
        line_1_vector = (0 - center_x, 0 - center_y)
        line_2_vector = (quantization - 1 - center_x, 0 - center_y)
        line_3_vector = (0 - center_x, quantization - 1 - center_y)
        line_1_normalized = (line_1_vector[0] / np.sqrt(line_1_vector[0]**2 + line_1_vector[1]**2), line_1_vector[1] / np.sqrt(line_1_vector[0]**2 + line_1_vector[1]**2))
        line_2_normalized = (line_2_vector[0] / np.sqrt(line_2_vector[0]**2 + line_2_vector[1]**2), line_2_vector[1] / np.sqrt(line_2_vector[0]**2 + line_2_vector[1]**2))
        line_3_normalized = (line_3_vector[0] / np.sqrt(line_3_vector[0]**2 + line_3_vector[1]**2), line_3_vector[1] / np.sqrt(line_3_vector[0]**2 + line_3_vector[1]**2))
        cv.line(hist_colored, (0, 0), (int(0 - line_1_normalized[0] * self.CORNER_LINE_LENGTH), int(0 - line_1_normalized[1] * self.CORNER_LINE_LENGTH)), (self.RGB_COLORMAP_BGR[0, 0]*255).astype(np.uint8).tolist(), self.CORNER_LINE_THICKNESS)
        cv.line(hist_colored, (quantization - 1, 0), (quantization - 1 - self.CORNER_LINE_LENGTH, int(0 - line_2_normalized[1] * self.CORNER_LINE_LENGTH)), (self.RGB_COLORMAP_BGR[0, quantization - 1]*255).astype(np.uint8).tolist(), self.CORNER_LINE_THICKNESS)
        cv.line(hist_colored, (0, quantization - 1), (int(0 - line_3_normalized[0] * self.CORNER_LINE_LENGTH), quantization - 1 - self.CORNER_LINE_LENGTH), (self.RGB_COLORMAP_BGR[quantization - 1, 0]*255).astype(np.uint8).tolist(), self.CORNER_LINE_THICKNESS)

        return hist_colored, hist

    def __rgb_chrominance(self, image):
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
    
    def __visualize_rgb_chroma(self):
        g_channel = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        r_channel = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        b_channel = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        for i in range(self.image_size):
            for j in range(self.image_size):
                r_c = (i / (self.image_size - 1))
                b_c = (j / (self.image_size - 1))
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