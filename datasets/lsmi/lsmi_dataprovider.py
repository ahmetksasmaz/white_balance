import cv2 as cv
import numpy as np
import os
import json
from ..dataprovider import DataProvider
from ..data import Data
from .configuration import *


def _build_mcc_mask(mcc_coord: dict, height: int, width: int) -> np.ndarray:
    # MCCCoord is annotated at 2x resolution (original sensor output).
    # Divide all coordinates by 2 to map them to the stored TIFF resolution.
    mask = np.ones((height, width), dtype=np.uint8)
    for quad in mcc_coord.values():
        pts = np.array(
            [[float(p[0]) / 2.0, float(p[1]) / 2.0] for p in quad],
            dtype=np.float32,
        )
        pts = pts.reshape((-1, 1, 2)).astype(np.int32)
        cv.fillPoly(mask, [pts], color=0)
    return mask.astype(bool)


class LSMIDataProvider(DataProvider):
    def __init__(self, override_dimensions=(-1, -1)):
        super().__init__(override_dimensions)
        self.metadatas = []

        for camera in ["galaxy", "nikon", "sony"]:
            image_directory = ROOT_DIRECTORY + "/" + camera
            with open(image_directory + "/" + METADATA_FILENAME, "r") as f:
                meta = json.load(f)
            for place, placeInfo in meta.items():
                if placeInfo["NumOfLights"] == 2:
                    self.data_names.append(image_directory + "/" + place + "/" + place + "_1." + IMAGE_EXTENSION)
                    self.data_names.append(image_directory + "/" + place + "/" + place + "_12." + IMAGE_EXTENSION)
                    self.metadatas.append({"illuminants": [placeInfo["Light1"]], "illuminant_map_path": None, "placeInfo": placeInfo})
                    self.metadatas.append({"illuminants": [placeInfo["Light1"], placeInfo["Light2"]], "illuminant_map_path": image_directory + "/" + place + "/" + place + "_12." + COEFF_EXTENSION, "placeInfo": placeInfo})
                elif placeInfo["NumOfLights"] == 3:
                    self.data_names.append(image_directory + "/" + place + "/" + place + "_1." + IMAGE_EXTENSION)
                    self.data_names.append(image_directory + "/" + place + "/" + place + "_12." + IMAGE_EXTENSION)
                    self.data_names.append(image_directory + "/" + place + "/" + place + "_13." + IMAGE_EXTENSION)
                    self.data_names.append(image_directory + "/" + place + "/" + place + "_123." + IMAGE_EXTENSION)
                    self.metadatas.append({"illuminants": [placeInfo["Light1"]], "illuminant_map_path": None, "placeInfo": placeInfo})
                    self.metadatas.append({"illuminants": [placeInfo["Light1"], placeInfo["Light2"]], "illuminant_map_path": image_directory + "/" + place + "/" + place + "_12." + COEFF_EXTENSION, "placeInfo": placeInfo})
                    self.metadatas.append({"illuminants": [placeInfo["Light1"], placeInfo["Light3"]], "illuminant_map_path": image_directory + "/" + place + "/" + place + "_13." + COEFF_EXTENSION, "placeInfo": placeInfo})
                    self.metadatas.append({"illuminants": [placeInfo["Light1"], placeInfo["Light2"], placeInfo["Light3"]], "illuminant_map_path": image_directory + "/" + place + "/" + place + "_123." + COEFF_EXTENSION, "placeInfo": placeInfo})

    def _construct_data(self, index):
        data = Data()
        image_path = self.data_names[index]
        parts = image_path.split("/")
        place_with_illuminations = parts[-1].split(".")[0]
        camera_model = parts[-3]
        data.set_image_name(camera_model + "_" + place_with_illuminations)

        raw_image = cv.imread(image_path, cv.IMREAD_UNCHANGED).astype(np.float32)

        if camera_model == "galaxy":
            BLACK_LEVEL, SATURATION_LEVEL = GALAXY_BLACK_LEVEL, GALAXY_SATURATION_LEVEL
        elif camera_model == "nikon":
            BLACK_LEVEL, SATURATION_LEVEL = NIKON_BLACK_LEVEL, NIKON_SATURATION_LEVEL
        else:
            BLACK_LEVEL, SATURATION_LEVEL = SONY_BLACK_LEVEL, SONY_SATURATION_LEVEL

        raw_image = np.clip((raw_image - BLACK_LEVEL) / (SATURATION_LEVEL - BLACK_LEVEL), 0, 1)
        data.set_quantization(SATURATION_LEVEL - BLACK_LEVEL)

        metadata = self.metadatas[index]
        mcc_coord = metadata["placeInfo"].get("MCCCoord", {})
        orig_h, orig_w = raw_image.shape[:2]
        mask = _build_mcc_mask(mcc_coord, orig_h, orig_w) if mcc_coord else np.ones((orig_h, orig_w), dtype=bool)

        if self.override_dimensions[0] > 0 and self.override_dimensions[1] > 0:
            raw_image = cv.resize(raw_image, self.override_dimensions)
            mask = cv.resize(mask.astype(np.uint8), self.override_dimensions, interpolation=cv.INTER_NEAREST).astype(bool)
        elif self.override_dimensions[0] > 0:
            aspect_ratio = raw_image.shape[1] / raw_image.shape[0]
            new_width = self.override_dimensions[0]
            new_size = (new_width, int(new_width / aspect_ratio))
            raw_image = cv.resize(raw_image, new_size)
            mask = cv.resize(mask.astype(np.uint8), new_size, interpolation=cv.INTER_NEAREST).astype(bool)
        elif self.override_dimensions[1] > 0:
            aspect_ratio = raw_image.shape[1] / raw_image.shape[0]
            new_height = self.override_dimensions[1]
            new_size = (int(new_height * aspect_ratio), new_height)
            raw_image = cv.resize(raw_image, new_size)
            mask = cv.resize(mask.astype(np.uint8), new_size, interpolation=cv.INTER_NEAREST).astype(bool)

        data.set_raw_image(raw_image)
        data.set_mask(mask)

        illuminants = []
        for illuminant in metadata["illuminants"]:
            illuminant_rgb = np.array(illuminant, dtype=np.float32)
            illuminant_chroma = illuminant_rgb / np.linalg.norm(illuminant_rgb)
            illuminants.append((illuminant_chroma[2] / illuminant_chroma[1], illuminant_chroma[0] / illuminant_chroma[1]))
        data.set_illuminants(illuminants)

        if metadata["illuminant_map_path"] is not None:
            coeff_map = np.load(metadata["illuminant_map_path"]).astype(np.float32)
            if coeff_map.ndim == 2:
                coeff_map = coeff_map[..., np.newaxis]
            illuminant_map = np.zeros((coeff_map.shape[0], coeff_map.shape[1], 2), dtype=np.float32)
            for i, illuminant in enumerate(illuminants):
                illuminant_map[..., 0] += coeff_map[..., i] * illuminant[0]
                illuminant_map[..., 1] += coeff_map[..., i] * illuminant[1]

            if self.override_dimensions[0] > 0 and self.override_dimensions[1] > 0:
                illuminant_map = cv.resize(illuminant_map, self.override_dimensions)
            elif self.override_dimensions[0] > 0:
                aspect_ratio = illuminant_map.shape[1] / illuminant_map.shape[0]
                new_width = self.override_dimensions[0]
                illuminant_map = cv.resize(illuminant_map, (new_width, int(new_width / aspect_ratio)))
            elif self.override_dimensions[1] > 0:
                aspect_ratio = illuminant_map.shape[1] / illuminant_map.shape[0]
                new_height = self.override_dimensions[1]
                illuminant_map = cv.resize(illuminant_map, (int(new_height * aspect_ratio), new_height))
            data.set_illuminant_map(illuminant_map)

        return data
