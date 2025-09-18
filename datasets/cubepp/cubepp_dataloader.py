from .configuration import ROOT_DIRECTORY, INPUT_IMAGE_DIRECTORY, IMAGE_EXTENSION
from .cubepp_data import CubePPData
import glob

class CubePPDataLoader:
    def __init__(self, load_resized=False):
        self.load_resized = load_resized
        self.image_names = sorted(glob.glob(INPUT_IMAGE_DIRECTORY+"/*."+IMAGE_EXTENSION))
        self.gt_lines = {}
        with open(ROOT_DIRECTORY+"/gt.csv", "r") as f:
            raw_lines = f.readlines()
            for line in raw_lines:
                line = line.strip()
                first_comma_index = line.index(",")
                image_name = line[:first_comma_index]
                self.gt_lines[image_name] = line[first_comma_index+1:]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if 0 <= index < len(self.image_names):
            image_path = self.image_names[index]
            image_name = image_path.split("/")[-1].split(".")[0]
            gt_line = self.gt_lines[image_name]
            return CubePPData(image_path, gt_line, load_resized=self.load_resized)
        else:
            raise IndexError("Image index out of range")