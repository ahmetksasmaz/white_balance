import cv2 as cv
import numpy as np
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *

class LSMIDataProvider(DataProvider):
    def __init__(self, override_dimensions=(-1, -1)):
        super().__init__()
        self.override_dimensions = override_dimensions
        self.galaxy_data_names = []
        self.nikon_data_names = []
        self.sony_data_names = []

        self.data_names = IMAGE_LIST
    def _construct_data(self, index):
        data = Data()

        return data