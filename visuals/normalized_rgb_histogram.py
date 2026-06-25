import numpy as np
import cv2 as cv
from visuals.visuals import Visuals
from helper import log_chrominance, histogram, pdf, prepare_display

class NormalizedRGBHistogram(Visuals):
    def __init__(self, image_size=512):
        super().__init__(image_size)

        self.COLOR_POWER = 1/2.8
        self.HIST_RANGE = [0, 1.0]
        self.CORNER_LINE_LENGTH = 25
        self.CORNER_LINE_THICKNESS = 2
        self.RGB_COLORMAP_BGR = self.__visualize_rgb_chroma()

    def _visualize(self, data):
        return self.__draw_rgb_chroma_from_image(data)

    def __draw_rgb_chroma_from_image(self, data):
        image = data.get_raw_image()
        input_b_c, input_g_c, input_r_c = self.__rgb_chrominance(image)

        bins = self.image_size
        hist, xedges, yedges = np.histogram2d(input_r_c.flatten(), input_b_c.flatten(), bins=[bins, bins], range=[self.HIST_RANGE, self.HIST_RANGE])

        normalized_hist = hist / (np.max(hist) + 1e-8)

        hist_colored = np.zeros((bins, bins, 3), dtype=np.float32)
        for i in range(bins):
            for j in range(bins):
                hist_colored[i, j] = self.RGB_COLORMAP_BGR[i, j] * normalized_hist[i, j]

        hist_colored = np.power(hist_colored, self.COLOR_POWER)
        max_val = np.max(hist_colored)
        if max_val > 0:
            hist_colored = hist_colored / max_val
        hist_colored = (hist_colored * 255.0).astype(np.uint8)
        cv.line(hist_colored, (0, bins - 1), (bins - 1, 0), (255, 255, 255), 3)

        center_x = (0 + 0 + bins - 1) // 3
        center_y = (0 + 0 + bins - 1) // 3
        line_1_vector = (0 - center_x, 0 - center_y)
        line_2_vector = (bins - 1 - center_x, 0 - center_y)
        line_3_vector = (0 - center_x, bins - 1 - center_y)
        line_1_normalized = (line_1_vector[0] / np.sqrt(line_1_vector[0]**2 + line_1_vector[1]**2), line_1_vector[1] / np.sqrt(line_1_vector[0]**2 + line_1_vector[1]**2))
        line_2_normalized = (line_2_vector[0] / np.sqrt(line_2_vector[0]**2 + line_2_vector[1]**2), line_2_vector[1] / np.sqrt(line_2_vector[0]**2 + line_2_vector[1]**2))
        line_3_normalized = (line_3_vector[0] / np.sqrt(line_3_vector[0]**2 + line_3_vector[1]**2), line_3_vector[1] / np.sqrt(line_3_vector[0]**2 + line_3_vector[1]**2))
        cv.line(hist_colored, (0, 0), (int(0 - line_1_normalized[0] * self.CORNER_LINE_LENGTH), int(0 - line_1_normalized[1] * self.CORNER_LINE_LENGTH)), (self.RGB_COLORMAP_BGR[0, 0]*255).astype(np.uint8).tolist(), self.CORNER_LINE_THICKNESS)
        cv.line(hist_colored, (bins - 1, 0), (bins - 1 - self.CORNER_LINE_LENGTH, int(0 - line_2_normalized[1] * self.CORNER_LINE_LENGTH)), (self.RGB_COLORMAP_BGR[0, bins - 1]*255).astype(np.uint8).tolist(), self.CORNER_LINE_THICKNESS)
        cv.line(hist_colored, (0, bins - 1), (int(0 - line_3_normalized[0] * self.CORNER_LINE_LENGTH), bins - 1 - self.CORNER_LINE_LENGTH), (self.RGB_COLORMAP_BGR[bins - 1, 0]*255).astype(np.uint8).tolist(), self.CORNER_LINE_THICKNESS)

        return hist_colored, hist

    def __rgb_chrominance(self, image):
        mask = (image[..., 0] != 0) | (image[..., 1] != 0) | (image[..., 2] != 0)
        masked_image = np.reshape(image[mask], (-1, 1, 3))
        b, g, r = cv.split(masked_image)
        sum_channels = r + g + b + 1e-8
        return b / sum_channels, g / sum_channels, r / sum_channels

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
                r_channel[i, j] = np.clip(r_c / sum_c, 0, 1)
                g_channel[i, j] = np.clip(g_c / sum_c, 0, 1)
                b_channel[i, j] = np.clip(b_c / sum_c, 0, 1)
        return cv.merge((b_channel, g_channel, r_channel))
