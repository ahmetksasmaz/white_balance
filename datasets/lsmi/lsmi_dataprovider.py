import cv2 as cv
import numpy as np
import os
import json
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *
import json

class LSMIDataProvider(DataProvider):
    def __init__(self, override_dimensions=(-1, -1)):
        super().__init__()
        self.override_dimensions = override_dimensions
        self.data_names = []
        self.metadatas = []

        for camera in ["galaxy", "nikon", "sony"]:
            image_directory = ROOT_DIRECTORY+"/"+camera

            with open(image_directory+"/"+METADATA_FILENAME, "r") as f:
                meta = json.load(f)
            
            for place, placeInfo in meta.items():
                if placeInfo["NumOfLights"] == 2:
                    self.data_names.append(image_directory+"/"+place+"/"+place+"_1."+IMAGE_EXTENSION)
                    self.data_names.append(image_directory+"/"+place+"/"+place+"_12."+IMAGE_EXTENSION)
                    self.metadatas.append({
                        "illuminants": [placeInfo["Light1"]],
                        "illuminant_map_path": None,
                        "placeInfo": placeInfo
                    })
                    self.metadatas.append({
                        "illuminants": [placeInfo["Light1"], placeInfo["Light2"]],
                        "illuminant_map_path": image_directory+"/"+place+"/"+place+"_12."+COEFF_EXTENSION,
                        "placeInfo": placeInfo
                    })
                elif placeInfo["NumOfLights"] == 3:
                    self.data_names.append(image_directory+"/"+place+"/"+place+"_1."+IMAGE_EXTENSION)
                    self.data_names.append(image_directory+"/"+place+"/"+place+"_12."+IMAGE_EXTENSION)
                    self.data_names.append(image_directory+"/"+place+"/"+place+"_13."+IMAGE_EXTENSION)
                    self.data_names.append(image_directory+"/"+place+"/"+place+"_123."+IMAGE_EXTENSION)
                    self.metadatas.append({
                        "illuminants": [placeInfo["Light1"]],
                        "illuminant_map_path": None,
                        "placeInfo": placeInfo
                    })
                    self.metadatas.append({
                        "illuminants": [placeInfo["Light1"], placeInfo["Light2"]],
                        "illuminant_map_path": image_directory+"/"+place+"/"+place+"_12."+COEFF_EXTENSION,
                        "placeInfo": placeInfo
                    })
                    self.metadatas.append({
                        "illuminants": [placeInfo["Light1"], placeInfo["Light3"]],
                        "illuminant_map_path": image_directory+"/"+place+"/"+place+"_13."+COEFF_EXTENSION,
                        "placeInfo": placeInfo
                    })
                    self.metadatas.append({
                        "illuminants": [placeInfo["Light1"], placeInfo["Light2"], placeInfo["Light3"]],
                        "illuminant_map_path": image_directory+"/"+place+"/"+place+"_123."+COEFF_EXTENSION,
                        "placeInfo": placeInfo
                    })

    def _construct_data(self, index):
        data = Data()
        image_path = self.data_names[index]
        image_path_splitted = image_path.split("/")
        place_with_illuminations = image_path_splitted[-1].split(".")[0]
        camera_model = image_path_splitted[-3]
        data.set_image_name(camera_model+"_"+place_with_illuminations)

        # Load Raw Image
        raw_image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        raw_image = raw_image.astype(np.float32)

        BLACK_LEVEL = None
        SATURATION_LEVEL = None
        if camera_model == "galaxy":
            BLACK_LEVEL = GALAXY_BLACK_LEVEL
            SATURATION_LEVEL = GALAXY_SATURATION_LEVEL
        elif camera_model == "nikon":
            BLACK_LEVEL = NIKON_BLACK_LEVEL
            SATURATION_LEVEL = NIKON_SATURATION_LEVEL
        elif camera_model == "sony":
            BLACK_LEVEL = SONY_BLACK_LEVEL
            SATURATION_LEVEL = SONY_SATURATION_LEVEL

        raw_image = np.clip((raw_image - BLACK_LEVEL) / (SATURATION_LEVEL - BLACK_LEVEL), 0, 1)
        if self.override_dimensions[0] > 0 and self.override_dimensions[1] > 0:
            raw_image = cv.resize(raw_image, self.override_dimensions)
        elif self.override_dimensions[0] > 0:
            aspect_ratio = raw_image.shape[1] / raw_image.shape[0]
            new_width = self.override_dimensions[0]
            new_height = int(new_width / aspect_ratio)
            raw_image = cv.resize(raw_image, (new_width, new_height))
        elif self.override_dimensions[1] > 0:
            aspect_ratio = raw_image.shape[1] / raw_image.shape[0]
            new_height = self.override_dimensions[1]
            new_width = int(new_height * aspect_ratio)
            raw_image = cv.resize(raw_image, (new_width, new_height))
        data.set_raw_image(raw_image)

        # Load GT info
        metadata = self.metadatas[index]
        metadata_illuminants = metadata["illuminants"]
        illuminants = []
        for illuminant in metadata_illuminants:
            illuminant_rgb = np.array(illuminant, dtype=np.float32)
            illuminant_chroma = illuminant_rgb / np.linalg.norm(illuminant_rgb)
            rg = illuminant_chroma[2] / illuminant_chroma[1]
            bg = illuminant_chroma[0] / illuminant_chroma[1]
            illuminants.append((rg, bg))
        data.set_illuminants(illuminants)

        if metadata["illuminant_map_path"] is not None:
            coeff_map = np.load(metadata["illuminant_map_path"])
            coeff_map = coeff_map.astype(np.float32)
            if self.override_dimensions[0] > 0 and self.override_dimensions[1] > 0:
                coeff_map = cv.resize(coeff_map, self.override_dimensions)
            elif self.override_dimensions[0] > 0:
                aspect_ratio = coeff_map.shape[1] / coeff_map.shape[0]
                new_width = self.override_dimensions[0]
                new_height = int(new_width / aspect_ratio)
                coeff_map = cv.resize(coeff_map, (new_width, new_height))
            elif self.override_dimensions[1] > 0:
                aspect_ratio = coeff_map.shape[1] / coeff_map.shape[0]
                new_height = self.override_dimensions[1]
                new_width = int(new_height * aspect_ratio)
                coeff_map = cv.resize(coeff_map, (new_width, new_height))
            data.set_illuminant_map(coeff_map)

        return data