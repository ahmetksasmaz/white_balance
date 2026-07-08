import os

import cv2 as cv
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from white_balance_algorithms.svwb_unet.external.model import U_Net
    from white_balance_algorithms.svwb_unet.external.utils import rgb2uvl, apply_wb
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    U_Net = None
    rgb2uvl = None
    apply_wb = None
    TORCH_AVAILABLE = False

from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm


class SVWBUnet(WhiteBalanceAlgorithm):
    requires_network = True
    def __init__(self, weights_dir=None, default_camera='galaxy', device=None):
        if not TORCH_AVAILABLE:
            raise ImportError(
                'SVWBUnet requires torch. Install it with `pip install torch` '
                'or remove svwb_unet:default from your configuration.'
            )
        self.device = torch.device(device if device is not None else 'cpu')
        self.model = U_Net(img_ch=3, output_ch=2)
        self.model.to(self.device)
        self.model.eval()

        if weights_dir is None:
            weights_dir = os.path.join(os.path.dirname(__file__), 'pretrained_weights')
        self.weights_dir = weights_dir

        self.camera_weights = {
            'galaxy': 'galaxy.pt',
            'nikon': 'nikon.pt',
            'sony': 'sony.pt',
        }
        self.default_camera = default_camera.lower()
        self.current_camera = None
        self._load_weights_for_camera(self.default_camera)

    def _get_weight_path(self, camera_name=None):
        if camera_name is None:
            camera_name = self.default_camera
        camera_name = str(camera_name).lower()
        filename = self.camera_weights.get(camera_name, self.camera_weights.get(self.default_camera, 'galaxy.pt'))
        return os.path.join(self.weights_dir, filename)

    def _load_weights_for_camera(self, camera_name=None):
        camera_name = camera_name or self.default_camera
        camera_name = str(camera_name).lower()
        if camera_name == self.current_camera:
            return
        weight_path = self._get_weight_path(camera_name)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f'SVWBUnet weight file not found: {weight_path}')
        checkpoint = torch.load(weight_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if any(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)
        self.current_camera = camera_name

    def _prepare_input(self, raw_bgr):
        image = np.asarray(raw_bgr, dtype=np.float32)
        if image.max() > 1.1:
            image = image / 255.0
        image_rgb = np.clip(image[..., ::-1], 1e-8, None)
        uvl = rgb2uvl(image_rgb)
        uvl = np.ascontiguousarray(np.transpose(uvl, (2, 0, 1)))
        return torch.from_numpy(uvl).unsqueeze(0).to(self.device)

    def _pad_to_multiple(self, tensor, multiple=256):
        _, _, h, w = tensor.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return tensor, (0, 0)
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate'), (pad_h, pad_w)

    def _unpad(self, tensor, pad_shape):
        pad_h, pad_w = pad_shape
        if pad_h == 0 and pad_w == 0:
            return tensor
        h = tensor.shape[2] - pad_h if pad_h != 0 else tensor.shape[2]
        w = tensor.shape[3] - pad_w if pad_w != 0 else tensor.shape[3]
        return tensor[:, :, :h, :w]

    def _estimate(self, data, process_masked=False):
        raw_image = data.get_raw_image()
        if raw_image is None:
            return {'single_illuminant': None, 'multi_illuminants': None, 'illuminant_map': None, 'estimated_srgb_image': None}

        camera = str(data.get_camera()).lower() if data.get_camera() is not None else self.default_camera
        self._load_weights_for_camera(camera)

        if process_masked and data.get_mask() is not None:
            mask = data.get_mask()
            if mask.shape != raw_image.shape[:2]:
                mask = cv.resize(mask.astype(np.uint8), (raw_image.shape[1], raw_image.shape[0]), interpolation=cv.INTER_NEAREST).astype(bool)
            raw_image = raw_image.copy()
            raw_image[~mask] = 0.0

        original_shape = raw_image.shape[:2]
        resized_raw = cv.resize(raw_image, (256, 256), interpolation=cv.INTER_AREA)
        if process_masked and data.get_mask() is not None:
            mask = data.get_mask()
            resized_mask = cv.resize(mask.astype(np.uint8), (256, 256), interpolation=cv.INTER_NEAREST).astype(bool)
            resized_raw = resized_raw.copy()
            resized_raw[~resized_mask] = 0.0

        # Prepare UVL input tensor (BGR → RGB → UVL)
        input_uvl = self._prepare_input(resized_raw)
        padded_input, pad_shape = self._pad_to_multiple(input_uvl, multiple=256)
        with torch.no_grad():
            prediction = self.model(padded_input)
        prediction = self._unpad(prediction, pad_shape)

        # Prepare input_rgb tensor (BGR → RGB, same scale) for apply_wb
        raw_rgb = np.ascontiguousarray(resized_raw[..., ::-1].astype(np.float32))
        if raw_rgb.max() > 1.1:
            raw_rgb = raw_rgb / 255.0
        raw_rgb = np.clip(raw_rgb, 1e-8, None)
        input_rgb_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(raw_rgb, (2, 0, 1)))
        ).unsqueeze(0).to(self.device)


        pred_r = torch.exp(padded_input[0,0,:] - prediction[:, 0]).squeeze(0).cpu().numpy()
        pred_b = torch.exp(padded_input[0,1,:] - prediction[:, 1]).squeeze(0).cpu().numpy()
        pred_g = np.ones_like(pred_r)

        pred_illum = np.stack([pred_b, pred_g, pred_r], axis=-1) * 255  # (H, W, 3)

        # # Apply white balance: R_wb = G * exp(pred[:,0]), B_wb = G * exp(pred[:,1])
        # b_channel = torch.exp(-prediction[:,0]).squeeze(0).cpu().numpy()
        # r_channel = torch.exp(-prediction[:,1]).squeeze(0).cpu().numpy()

        # illuminant_map (H, W, 2): [R_illuminant, B_illuminant]
        illuminant_map = np.stack([pred_b, pred_r], axis=-1)  # (H, W, 2)
        illuminant_map = cv.resize(illuminant_map, (original_shape[1], original_shape[0]), interpolation=cv.INTER_LINEAR)

        return {'single_illuminant': None, 'multi_illuminants': None, 'illuminant_map': illuminant_map, 'estimated_srgb_image': None}
