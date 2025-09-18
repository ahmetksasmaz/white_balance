"""
 Deep white-balance editing main function (inference phase)
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import numpy as np
import torch
from torchvision import transforms
import utilities.utils as utls
import cv2 as cv


def deep_wb(image, task='all', net_awb=None, net_t=None, net_s=None, device='cpu', s=656):
    # check image size
    h, w, _ = image.shape
    image_resized = cv.resize(image, (round(w / max(h, w) * s), round(h / max(h, w) * s)))
    h, w, _ = image_resized.shape
    if w % 2 ** 4 == 0:
        new_size_w = w
    else:
        new_size_w = w + 2 ** 4 - w % 2 ** 4

    if h % 2 ** 4 == 0:
        new_size_h = h
    else:
        new_size_h = h + 2 ** 4 - h % 2 ** 4

    inSz = (new_size_w, new_size_h)
    if not ((w, h) == inSz):
        image_resized = cv.resize(image_resized, inSz)

    img = np.transpose(image_resized, (2, 0, 1)) # HWC to CHW
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)

    net_awb.eval()
    with torch.no_grad():
        output_awb = net_awb(img)

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    output_awb = tf(torch.squeeze(output_awb.cpu()))
    output_awb = output_awb.squeeze().cpu().numpy()
    output_awb = output_awb.transpose((1, 2, 0))
    m_awb = utls.get_mapping_func(image_resized, output_awb)
    output_awb = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_awb))

    return output_awb
