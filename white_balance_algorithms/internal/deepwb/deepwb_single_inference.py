import torch
from PIL import Image
from utilities.deepWB import deep_wb
from arch import deep_wb_single_task
import numpy as np
import cv2 as cv

class DeepWBSingleInference:
    def __init__(self, model_path="models/net_awb.pth", max_dim=656):
        self.max_dim = max_dim
        self.net_awb = deep_wb_single_task.deepWBnet()
        self.net_awb.load_state_dict(torch.load(model_path))
        self.net_awb.eval()

    def infer_with_path(self, image_path):
        # img = Image.open(image_path)
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        out_awb = deep_wb(img, task="awb", net_awb=self.net_awb, device="cpu", s=self.max_dim)
        return out_awb
    def infer(self, image):
        rgb_image = image[...,::-1]
        out_awb = deep_wb(rgb_image, task="awb", net_awb=self.net_awb, device="cpu", s=self.max_dim)
        return out_awb

if __name__ == "__main__":
    model_path = "models/net_awb.pth"
    image_path = "../../external/deepwb/example_images/04.JPG"

    deepwb_inference = DeepWBSingleInference(model_path=model_path)


    img = cv.imread(image_path)
    img = img.astype(np.float32) / 255.0
    result_image = deepwb_inference.infer(img)
    result_image = np.array(result_image)
    result_image = (result_image * 255).astype(np.uint8)
    result_image = cv.cvtColor(result_image, cv.COLOR_RGB2BGR)
    cv.imshow("Result Image", result_image)
    cv.waitKey(0)