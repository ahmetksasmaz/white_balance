import sys
import cv2 as cv
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from helpers import *

from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout, QHBoxLayout, QGroupBox
)

COLUMN_HEADERS = [
    "corrected image", "input image", "gt image", "masked input image",
    "inverse masked input image", "masked gt image", "inverse masked gt image"
]

def cvimg_to_qpixmap(img):
    if img.ndim == 2:
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 1, QImage.Format_Grayscale8)
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class GridViewer(QWidget):
    def __init__(self, images):
        super().__init__()
        layout = QGridLayout()
        self.input_image = images[0][0].astype(np.float32) / 255.0
        self.input_mean = np.mean(self.input_image, axis=(0,1))
        self.gain = np.array([self.input_mean[1], self.input_mean[1], self.input_mean[1]]) / self.input_mean
        # Set column headers
        for col, header in enumerate(COLUMN_HEADERS):
            if col == 1:
                header = "input image\n(mean RGB: {:.2f}, {:.2f}, {:.2f})\n(Gain: {:.2f}, {:.2f}, {:.2f})".format(
                    self.input_mean[2], self.input_mean[1], self.input_mean[0],
                    1/self.gain[2], 1/self.gain[1], 1/self.gain[0]
                )
            label = QLabel(header)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label, 0, col+1)

        # Fill cells with images
        for row in range(2):
            for col in range(len(COLUMN_HEADERS)):
                if col == 0:
                    continue
                img = images[row][col-1]
                img_label = QLabel()
                img_label.setAlignment(Qt.AlignCenter)
                img_label.setPixmap(cvimg_to_qpixmap(img).scaled(256, 256, Qt.KeepAspectRatio))
                layout.addWidget(img_label, row + 1, col+1)
        self.setLayout(layout)

        self.set_input_image(images[0][0])
        self.set_histogram_image(images[1][0])

    def set_input_image(self, img):
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setPixmap(cvimg_to_qpixmap(img).scaled(256, 256, Qt.KeepAspectRatio))
        if self.layout().itemAtPosition(0, 0):
            widget = self.layout().itemAtPosition(0, 0).widget()
            if widget:
                self.layout().removeWidget(widget)
                widget.deleteLater()
        self.layout().addWidget(img_label, 1, 1)
    def set_histogram_image(self, img):
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setPixmap(cvimg_to_qpixmap(img).scaled(256, 256, Qt.KeepAspectRatio))
        if self.layout().itemAtPosition(0, 0):
            widget = self.layout().itemAtPosition(0, 0).widget()
            if widget:
                self.layout().removeWidget(widget)
                widget.deleteLater()
        self.layout().addWidget(img_label, 2, 1)
    def set_illumination(self, b, g, r):
        illuminant = np.array([b, g, r], dtype=np.float32)
        gain = np.array([illuminant[1], illuminant[1], illuminant[1]]) / illuminant
        adjusted_image = self.input_image * gain
        adjusted_image = np.clip(adjusted_image, 0, 1)
        self.set_input_image((adjusted_image * 255).astype(np.uint8))

        adjusted_hist, _ = draw_log_chroma_from_image(adjusted_image)
        self.set_histogram_image(adjusted_hist) 


class ControlPanel(QWidget):
    def __init__(self, on_left, on_right, on_slider, image_label):
        super().__init__()
        layout = QHBoxLayout()
        # Left button
        self.left_btn = QPushButton("Left")
        layout.addWidget(self.left_btn)
        # Image label
        self.image_label = QLabel(image_label)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        # Right button
        self.right_btn = QPushButton("Right")
        layout.addWidget(self.right_btn)
        # Sliders for R, G, B with min/max/current value labels
        self.sliders = []
        for color in ["R", "G", "B"]:
            group = QGroupBox(f"{color} Illumination")
            group_layout = QVBoxLayout()
            slider_layout = QHBoxLayout()
            min_label = QLabel("0.0")
            max_label = QLabel("1.0")
            curr_label = QLabel("1.0")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(100)
            slider.setSingleStep(1)
            slider.valueChanged.connect(
                lambda val, c=color, cl=curr_label: (cl.setText(f"{val/100:.2f}"), on_slider(c, val / 100.0))
            )
            slider_layout.addWidget(min_label)
            slider_layout.addWidget(slider)
            slider_layout.addWidget(max_label)
            slider_layout.addWidget(curr_label)
            group_layout.addLayout(slider_layout)
            group.setLayout(group_layout)
            layout.addWidget(group)
            self.sliders.append(slider)
        self.left_btn.clicked.connect(on_left)
        self.right_btn.clicked.connect(on_right)
        self.setLayout(layout)
    def reset_sliders(self):
        for slider in self.sliders:
            slider.setValue(100)

class MainWindow(QWidget):
    def __init__(self, images_list):
        super().__init__()
        self.images_list = images_list
        self.current_idx = 0
        self.r = self.g = self.b = 1.0
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.grid_viewer = GridViewer(self.images_list[self.current_idx])
        self.control_panel = ControlPanel(
            self.on_left, self.on_right, self.on_slider,
            f"Image {self.current_idx + 1}/{len(self.images_list)}"
        )
        main_layout.addWidget(self.grid_viewer, 3)
        main_layout.addWidget(self.control_panel, 1)
        self.setLayout(main_layout)
        self.setWindowTitle("Vegetation Trials GUI")
        self.resize(1200, 500)

    def update_view(self):
        # Update grid viewer and image label
        self.layout().removeWidget(self.grid_viewer)
        self.grid_viewer.deleteLater()
        self.grid_viewer = GridViewer(self.images_list[self.current_idx])
        self.layout().insertWidget(0, self.grid_viewer, 3)
        self.control_panel.image_label.setText(f"Image {self.current_idx + 1}/{len(self.images_list)}")

    def on_left(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_view()
            self.r = self.g = self.b = 1.0
            self.control_panel.reset_sliders()

    def on_right(self):
        if self.current_idx < len(self.images_list) - 1:
            self.current_idx += 1
            self.update_view()
            self.r = self.g = self.b = 1.0
            self.control_panel.reset_sliders()

    def on_slider(self, color, value):
        if color == "R":
            self.r = value
        elif color == "G":
            self.g = value
        elif color == "B":
            self.b = value
        self.grid_viewer.set_illumination(self.b, self.g, self.r)
        # You can add illumination adjustment logic here

if __name__ == "__main__":
    app = QApplication(sys.argv)
    numbers = [4, 11, 13, 16, 22, 28, 37, 44, 49, 57]
    input_filenames = [f"images/00_{num:04d}.png" for num in numbers]
    gt_filenames = [f"images/00_{num:04d}_gt.png" for num in numbers]
    mask_filenames = [f"images/00_{num:04d}_gt.jpg" for num in numbers]

    images_list = []
    for i in range(len(input_filenames)):
        input_image = cv.imread(input_filenames[i]).astype(np.float32) / 255.0
        gt_image = cv.imread(gt_filenames[i]).astype(np.float32) / 255.0
        mask_image = cv.imread(mask_filenames[i], cv.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask_image = (mask_image > 0.5).astype(np.float32)  # Binary mask
        mask_image = cv.merge([mask_image, mask_image, mask_image])  # Convert to 3-channel
        masked_input = input_image * mask_image
        masked_gt = gt_image * mask_image
        inverse_mask = 1.0 - mask_image
        inverse_masked_input = input_image * inverse_mask
        inverse_masked_gt = gt_image * inverse_mask

        input_hist, _ = draw_log_chroma_from_image(input_image)
        gt_hist, _ = draw_log_chroma_from_image(gt_image)
        masked_input_hist, _ = draw_log_chroma_from_image(masked_input)
        inverse_masked_input_hist, _ = draw_log_chroma_from_image(inverse_masked_input)
        masked_gt_hist, _ = draw_log_chroma_from_image(masked_gt)
        inverse_masked_gt_hist, _ = draw_log_chroma_from_image(inverse_masked_gt)

        images_list.append([
            [(input_image * 255).astype(np.uint8),
            (gt_image * 255).astype(np.uint8),
            (masked_input * 255).astype(np.uint8),
            (inverse_masked_input * 255).astype(np.uint8),
            (masked_gt * 255).astype(np.uint8),
            (inverse_masked_gt * 255).astype(np.uint8),],
            [input_hist,
            gt_hist,
            masked_input_hist,
            inverse_masked_input_hist,
            masked_gt_hist,
            inverse_masked_gt_hist]
        ])
    
    window = MainWindow(images_list)
    window.show()
    sys.exit(app.exec_())