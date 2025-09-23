from .configuration import *
from .lsmi_data import LSMIData
import json

class LSMIDataLoader:
    def __init__(self, load_resized=False):
        self.load_resized = load_resized
        self.camera_models = ["galaxy", "nikon", "sony"]
        self.image_coeff_lights = []

        for camera_model in self.camera_models:
            if camera_model == "galaxy":
                image_directory = GALAXY_IMAGE_DIRECTORY
            elif camera_model == "nikon":
                image_directory = NIKON_IMAGE_DIRECTORY
            elif camera_model == "sony":
                image_directory = SONY_IMAGE_DIRECTORY

            with open(image_directory+"/"+METADATA_FILENAME, "r") as f:
                meta = json.load(f)

            for place, placeInfo in meta.items():
                lights = []
                if placeInfo["NumOfLights"] == 2:
                    lights.append(placeInfo["Light1"])
                    lights.append(placeInfo["Light2"])
                elif placeInfo["NumOfLights"] == 3:
                    lights.append(placeInfo["Light1"])
                    lights.append(placeInfo["Light2"])
                    lights.append(placeInfo["Light3"])
                postfix = "12"
                if len(lights) == 3:
                    postfix = "123"
                image_path = image_directory+"/"+place+"/"+place+"_"+postfix+"."+IMAGE_EXTENSION
                coeff_path = image_directory+"/"+place+"/"+place+"_"+postfix+"."+COEFF_EXTENSION
                self.image_coeff_lights.append((image_path, coeff_path, lights))
    def __len__(self):
        return len(self.image_coeff_lights)
    
    def __getitem__(self, index):
        if 0 <= index < len(self.image_coeff_lights):
            image_path, coeff_path, lights = self.image_coeff_lights[index]
            return LSMIData(image_path, coeff_path, lights, load_resized=self.load_resized)
        else:
            raise IndexError("Image index out of range")