import cv2 as cv
import numpy as np
import os
import json
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *

class LSMIDataProvider(DataProvider):
    def __init__(self, directories, override_dimensions=(-1, -1)):
        super().__init__()
        self.directories = directories
        self.override_dimensions = override_dimensions
        self.data_names = []
        self.metadata = {}
        self.coefficient_maps = {}

        for directory in self.directories:
            self.data_names.extend([f"{place}_{illuminants}.tiff" for place in os.listdir(directory) if place.startswith("Place")])
            metadata_path = os.path.join(directory, "metadata.csv")
            with open(metadata_path, "r") as f:
                self.metadata[directory] = [line.strip().split(",") for line in f.readlines()[1:]]
            coeff_map_path = os.path.join(directory, "coeff_map.npy")
            self.coefficient_maps[directory] = np.load(coeff_map_path)

    def _construct_data(self, index):
        data = Data()
        image_name = self.data_names[index]
        directory = next(directory for directory in self.directories if image_name in os.listdir(directory))
        place, illuminants = image_name.split("_")
        illuminants = int(illuminants.split(".")[0])

        # Load image
        image_path = os.path.join(directory, image_name)
        image = self._load_image(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        # Normalize image
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)

        # Resize image if necessary
        if self.override_dimensions != (-1, -1):
            image = cv.resize(image, self.override_dimensions)

        # Load metadata
        metadata = self.metadata[directory][int(place[4:]) - 1]

        # Extract illuminant chromaticities
        illuminant_map = {metadata[i][0]: (float(metadata[i][1]), float(metadata[i][2])) for i in range(1, illuminants + 1)}

        # Set exposure values
        exposure_values = {
            "exposure_time": float(metadata[illuminants + 1]),
            "iso": float(metadata[illuminants + 2]),
            "aperture": float(metadata[illuminants + 3])
        }

        # Set data attributes
        data.set_image_name(image_name)
        data.set_raw_image(image)
        data.set_illuminant_map(illuminant_map)
        data.set_scene_data(metadata)
        data.set_exposure_values(exposure_values["exposure_time"], exposure_values["iso"], exposure_values["aperture"])

        return data

    def _load_image(self, image_path):
        image = cv.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")
        return image
