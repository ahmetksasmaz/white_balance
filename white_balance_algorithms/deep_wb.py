import numpy as np
import cv2 as cv
from .white_balance_algorithm import WhiteBalanceAlgorithm
from chromatic_adaptation import WhitePoint
from .internal.deepwb.deepwb_single_inference import DeepWBSingleInference
from helper import gamma_correction, prepare_display

class DeepWB(WhiteBalanceAlgorithm):
    def __init__(self, model_path: str, max_dim: int = 656):
        self.model_path = model_path
        self.max_dim = max_dim
        self.deepwb_inference = DeepWBSingleInference(model_path=self.model_path, max_dim=self.max_dim)

    def _apply_internal(self, image):
        image = image[...,::-1]
        gamma_corrected_image = gamma_correction(image, gamma=2.2)
        inference = self.deepwb_inference.infer(gamma_corrected_image)
        inverse_gamma_corrected_image = gamma_correction(inference, gamma=1/2.2)
        inverse_gamma_corrected_image = inverse_gamma_corrected_image[...,::-1]
        return inverse_gamma_corrected_image, {}