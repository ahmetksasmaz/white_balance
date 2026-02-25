import cv2 as cv
import numpy as np
import os
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *

class LSMIDataProvider(DataProvider):
    def __init__(self, override_dimensions=(-1, -1)):
        super().__init__()
        self.override_dimensions = override_dimensions
        self.data_names = IMAGE_LIST

    def _construct_data(self, index):
        data = Data()
        image_name = self.data_names[index]
        image_path = os.path.join(GALAXY_IMAGE_DIRECTORY, image_name)
        coeff_path = os.path.join(GALAXY_COEFF_DIRECTORY, image_name.replace(".png", ".json"))

        # Load image
        image = cv.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        # Normalize image
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)

        # Resize image if necessary
        if self.override_dimensions != (-1, -1):
            image = cv.resize(image, self.override_dimensions)

        # Load metadata
        with open(coeff_path, "r") as f:
            metadata = json.load(f)

        # Extract illuminant chromaticities
        illuminants = metadata["illuminants"]
        illuminant_map = {ill["name"]: (ill["r"], ill["b"]) for ill in illuminants}

        # Set exposure values
        exposure_values = metadata["exposure_values"]

        # Set data attributes
        data.set_image_name(image_name)
        data.set_raw_image(image)
        data.set_illuminant_map(illuminant_map)
        data.set_scene_data(metadata)
        data.set_exposure_values(exposure_values["exposure_time"], exposure_values["iso"], exposure_values["aperture"])

        return data
