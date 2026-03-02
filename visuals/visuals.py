import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

class Visuals:
    def __init__(self, image_size=512):
        self.image_size = image_size
    
    def visualize(self, data):
        self._visualize(data)
    
    def _visualize(self, data):
        raise NotImplementedError("Subclasses must implement this method")