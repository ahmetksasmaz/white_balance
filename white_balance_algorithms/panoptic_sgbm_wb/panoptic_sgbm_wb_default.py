import os
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

try:
    import torch
    from sklearn.decomposition import PCA
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    TORCH_AVAILABLE = True
    SAM_AVAILABLE = True
except ImportError:
    torch = None
    PCA = None
    SamAutomaticMaskGenerator = None
    sam_model_registry = None
    TORCH_AVAILABLE = False
    SAM_AVAILABLE = False

from datasets.data import Data
from white_balance_algorithms.white_balance_algorithm import WhiteBalanceAlgorithm

DEFAULT_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")
DEFAULT_CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
DEFAULT_CHECKPOINT = os.environ.get(
    "SAM_CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "pretrained_weights", DEFAULT_CHECKPOINT_NAME)
)
EPS = 1e-6
SAM_CHECKPOINT_URLS = {
    DEFAULT_CHECKPOINT_NAME: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}


def download_checkpoint(dest_path: str, url: str) -> None:
    from urllib.request import urlretrieve
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"Downloading SAM checkpoint from {url} to {dest_path}...")
    urlretrieve(url, dest_path)
    print("SAM checkpoint download complete.")


def resize_for_inference(img: np.ndarray, max_side: int = 1024):
    height, width = img.shape[:2]
    if max(height, width) <= max_side:
        return img, (height, width)
    scale = max_side / max(height, width)
    new_w = max(32, ((int(round(width * scale)) + 31) // 32) * 32)
    new_h = max(32, ((int(round(height * scale)) + 31) // 32) * 32)
    return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR), (new_h, new_w)


def resize_for_sgbm(img: np.ndarray, max_width: int = 300, max_height: int = 200):
    height, width = img.shape[:2]
    if width <= max_width and height <= max_height:
        return img, (height, width)
    scale = min(max_width / width, max_height / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR), (new_h, new_w)


def merge_masks(masks, height: int, width: int):
    best_score = np.zeros((height, width), dtype=np.float32)
    best_instance = np.full((height, width), -1, dtype=np.int32)
    for idx, mask_data in enumerate(masks):
        mask = mask_data["segmentation"].astype(np.uint8)
        if mask.shape[:2] != (height, width):
            mask = cv.resize(mask, (width, height), interpolation=cv.INTER_NEAREST)
        score = float(mask_data.get("predicted_iou", 0.0))
        mask_bool = mask > 0
        update_pixels = mask_bool & (score > best_score)
        if np.any(update_pixels):
            best_score[update_pixels] = score
            best_instance[update_pixels] = idx
    return best_instance


def log_chromaticity(img_rgb: np.ndarray) -> np.ndarray:
    img = np.clip(img_rgb, EPS, None)
    log_rg = np.log(img[..., 0] + EPS) - np.log(img[..., 1] + EPS)
    log_gb = np.log(img[..., 1] + EPS) - np.log(img[..., 2] + EPS)
    return np.stack((log_rg, log_gb), axis=-1)


def normalize_vector_field(vectors: np.ndarray) -> np.ndarray:
    norm = np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), EPS)
    return vectors / norm


def make_dummy_pca():
    class DummyPCA:
        def __init__(self):
            self.components_ = np.zeros((1, 2), dtype=np.float32)
            self.mean_ = np.zeros(2, dtype=np.float32)

        def fit(self, X):
            return self

        def fit_transform(self, X):
            return np.zeros((X.shape[0], 1), dtype=np.float32)

        def transform(self, X):
            return np.zeros((X.shape[0], 1), dtype=np.float32)

        def inverse_transform(self, X):
            X = np.asarray(X, dtype=np.float32)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.zeros((X.shape[0], 2), dtype=np.float32)

    return DummyPCA()


def create_label_space(dtm_field: np.ndarray, num_labels: int = 16):
    flat_dtm = np.clip(np.nan_to_num(dtm_field.reshape(-1, 2), nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    if flat_dtm.size == 0 or np.allclose(flat_dtm, flat_dtm[0], atol=1e-9) or np.allclose(np.var(flat_dtm, axis=0), 0.0, atol=1e-12):
        return np.zeros(num_labels, dtype=np.float32), make_dummy_pca()
    pca = PCA(n_components=1)
    projected = np.nan_to_num(pca.fit_transform(flat_dtm), nan=0.0, posinf=0.0, neginf=0.0)
    min_val, max_val = projected.min(), projected.max()
    labels_1d = np.zeros(num_labels, dtype=np.float32) if np.isclose(min_val, max_val, atol=1e-12) else np.linspace(min_val, max_val, num_labels).astype(np.float32)
    return labels_1d, pca


def compute_data_cost(dtm_field: np.ndarray, labels_1d: np.ndarray, pca) -> np.ndarray:
    height, width, _ = dtm_field.shape
    flat_dtm = np.clip(np.nan_to_num(dtm_field.reshape(-1, 2), nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    projected_dtm = np.nan_to_num(pca.transform(flat_dtm), nan=0.0, posinf=0.0, neginf=0.0).reshape((height, width))
    return ((projected_dtm[:, :, np.newaxis] - labels_1d[np.newaxis, np.newaxis, :]) ** 2).astype(np.float32)


def _sgbm_step(data_cost_slice, prev_slice, P2_vec, P1_base):
    min_prev = prev_slice.min(axis=-1, keepdims=True)
    shift_up = np.empty_like(prev_slice)
    shift_up[:, 0] = np.inf
    shift_up[:, 1:] = prev_slice[:, :-1] + P1_base
    shift_dn = np.empty_like(prev_slice)
    shift_dn[:, -1] = np.inf
    shift_dn[:, :-1] = prev_slice[:, 1:] + P1_base
    best = np.minimum(np.minimum(prev_slice, shift_up), np.minimum(shift_dn, min_prev + P2_vec))
    return data_cost_slice + best - min_prev


def compute_path_cost(data_cost: np.ndarray, mask: np.ndarray, direction: Tuple[int, int], P1_base: float, P2_base: float) -> np.ndarray:
    height, width, _ = data_cost.shape
    dx, dy = direction
    L = data_cost.copy()

    def _p2_vec(curr, prev):
        return np.where(curr == prev, P2_base, P2_base * 0.1)[:, np.newaxis]

    if dx != 0 and dy == 0:
        for x in (range(1, height) if dx > 0 else range(height - 2, -1, -1)):
            px = x - dx
            L[x] = _sgbm_step(data_cost[x], L[px], _p2_vec(mask[x], mask[px]), P1_base)
    elif dy != 0 and dx == 0:
        for y in (range(1, width) if dy > 0 else range(width - 2, -1, -1)):
            py = y - dy
            L[:, y] = _sgbm_step(data_cost[:, y], L[:, py], _p2_vec(mask[:, y], mask[:, py]), P1_base)
    else:
        y_vals = np.arange(1, width) if dy > 0 else np.arange(0, width - 1)
        py_vals = y_vals - dy
        for x in (range(1, height) if dx > 0 else range(height - 2, -1, -1)):
            px = x - dx
            L[x, y_vals] = _sgbm_step(data_cost[x, y_vals], L[px, py_vals], _p2_vec(mask[x, y_vals], mask[px, py_vals]), P1_base)

    return L


def semi_global_matching(data_cost: np.ndarray, mask: np.ndarray, direction_count: int = 4, P1: float = 0.05, P2: float = 0.8) -> np.ndarray:
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] if direction_count == 4 else [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    S = np.zeros_like(data_cost, dtype=np.float32)
    for direction in directions:
        S += compute_path_cost(data_cost, mask, direction, P1, P2)
    return S


def winner_take_all(aggregated_cost: np.ndarray, labels_1d: np.ndarray, pca) -> np.ndarray:
    best_label_indices = np.argmin(aggregated_cost, axis=2)
    illuminant_map_1d = labels_1d[best_label_indices]
    height, width = illuminant_map_1d.shape
    return pca.inverse_transform(illuminant_map_1d.reshape(-1, 1)).reshape((height, width, 2))


def convert_log_chroma_to_linear(illuminant_map_log: np.ndarray) -> np.ndarray:
    return np.stack((np.exp(illuminant_map_log[..., 0]), np.exp(-illuminant_map_log[..., 1])), axis=-1)


def _resize_illuminant_map(illuminant_map_log: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if illuminant_map_log.ndim == 2:
        illuminant_map_log = np.stack((illuminant_map_log, np.zeros_like(illuminant_map_log)), axis=-1)
    if illuminant_map_log.shape[:2] != target_shape:
        illuminant_map_log = cv.resize(illuminant_map_log, (target_shape[1], target_shape[0]), interpolation=cv.INTER_LINEAR)
    if illuminant_map_log.shape[:2] != target_shape and illuminant_map_log.shape[1::-1] == target_shape:
        illuminant_map_log = np.transpose(illuminant_map_log, (1, 0, 2))
    if illuminant_map_log.shape[:2] != target_shape:
        illuminant_map_log = np.broadcast_to(
            np.nan_to_num(illuminant_map_log, nan=0.0, posinf=0.0, neginf=0.0),
            target_shape + (illuminant_map_log.shape[2],),
        )
    return np.nan_to_num(illuminant_map_log, nan=0.0, posinf=0.0, neginf=0.0)


def apply_sgbm_white_balance(raw_bgr: np.ndarray, illuminant_map_log: np.ndarray) -> np.ndarray:
    illuminant_map_log = _resize_illuminant_map(illuminant_map_log, raw_bgr.shape[:2])
    r_g = np.exp(illuminant_map_log[..., 0])
    b_g = np.exp(-illuminant_map_log[..., 1])
    gain_r = 1.0 / np.maximum(r_g, EPS)
    gain_g = np.ones_like(gain_r, dtype=np.float32)
    gain_b = 1.0 / np.maximum(b_g, EPS)
    gain_geo = np.cbrt(gain_r * gain_g * gain_b)
    inv_map = np.stack((gain_b / gain_geo, gain_g / gain_geo, gain_r / gain_geo), axis=-1)
    return np.clip(raw_bgr.astype(np.float32) * inv_map, 0.0, 1.0)


class PanopticSGBMWB(WhiteBalanceAlgorithm):
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: Optional[str] = None,
        device: Optional[str] = None,
        max_side: int = 1024,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.2,
        stability_score_thresh: float = 0.2,
        crop_n_layers: int = 0,
        min_mask_region_area: int = 1000,
        sgbm_directions: int = 4,
        sgbm_max_width: int = 300,
        sgbm_max_height: int = 200,
        num_labels: int = 16,
    ):
        if not TORCH_AVAILABLE or not SAM_AVAILABLE:
            raise ImportError('PanopticSGBMWB requires torch and segment-anything. Install them with `pip install torch segment-anything`.')

        self.model_type = model_type or DEFAULT_MODEL_TYPE
        self.checkpoint = checkpoint or DEFAULT_CHECKPOINT
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_side = max_side
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.crop_n_layers = crop_n_layers
        self.min_mask_region_area = min_mask_region_area
        self.sgbm_directions = sgbm_directions
        self.sgbm_max_width = sgbm_max_width
        self.sgbm_max_height = sgbm_max_height
        self.num_labels = num_labels

        self.sam = None
        self.generator = None
        self._load_sam_model()
        self._build_generator()

    def _load_sam_model(self):
        if not os.path.exists(self.checkpoint):
            basename = os.path.basename(self.checkpoint)
            if basename in SAM_CHECKPOINT_URLS:
                download_checkpoint(self.checkpoint, SAM_CHECKPOINT_URLS[basename])
            else:
                raise FileNotFoundError(f"SAM checkpoint not found: {self.checkpoint}")
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device=self.device)
        if self.device.startswith("cuda"):
            self.sam.to(dtype=torch.float16)

    def _build_generator(self):
        self.generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            crop_n_layers=self.crop_n_layers,
            min_mask_region_area=self.min_mask_region_area,
        )

    def _prepare_image(self, raw_image: np.ndarray) -> np.ndarray:
        image = np.asarray(raw_image, dtype=np.float32)
        if image.max() > 1.1:
            image = image / 255.0
        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        return image[..., ::-1]

    def _estimate(self, data: Data, process_masked: bool = False):
        raw_image = data.get_raw_image()
        if raw_image is None:
            return {"single_illuminant": None, "multi_illuminants": None, "illuminant_map": None, "estimated_srgb_image": None}

        raw_bgr = np.asarray(raw_image, dtype=np.float32)
        rgb_image = self._prepare_image(raw_bgr)
        height, width = rgb_image.shape[:2]

        infer_img, _ = resize_for_inference(rgb_image, max_side=self.max_side)
        masks = self.generator.generate(infer_img)
        if len(masks) == 0:
            return {"single_illuminant": None, "multi_illuminants": None, "illuminant_map": None, "estimated_srgb_image": None}

        best_instance = merge_masks(masks, height, width)
        sgbm_img, _ = resize_for_sgbm(rgb_image, max_width=self.sgbm_max_width, max_height=self.sgbm_max_height)
        log_chroma_small = log_chromaticity(sgbm_img)
        resized_instance = cv.resize(best_instance.astype(np.int32), (sgbm_img.shape[1], sgbm_img.shape[0]), interpolation=cv.INTER_NEAREST)
        illuminant_map_log_small = self._run_sgbm_color_constancy(log_chroma_small, resized_instance)
        illuminant_map_log = cv.resize(illuminant_map_log_small, (width, height), interpolation=cv.INTER_LINEAR)
        illuminant_map_linear = convert_log_chroma_to_linear(illuminant_map_log)
        estimated_srgb_image = apply_sgbm_white_balance(raw_bgr, illuminant_map_log)

        return {"single_illuminant": None, "multi_illuminants": None, "illuminant_map": illuminant_map_linear, "estimated_srgb_image": estimated_srgb_image}

    def _run_sgbm_color_constancy(self, log_chroma_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        dtm_field = self._compute_directional_dtm(log_chroma_img, mask)
        labels_1d, pca = create_label_space(dtm_field, num_labels=self.num_labels)
        data_cost = compute_data_cost(dtm_field, labels_1d, pca)
        aggregated_cost = semi_global_matching(data_cost, mask, direction_count=self.sgbm_directions, P1=0.02, P2=0.5)
        return winner_take_all(aggregated_cost, labels_1d, pca)

    def _compute_directional_dtm(self, log_chroma_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        dtm_field = np.zeros_like(log_chroma_img)
        for m_id in np.unique(mask):
            if m_id < 0:
                continue
            mask_indices = mask == m_id
            values = log_chroma_img[mask_indices]
            if values.size == 0:
                continue
            dtm_field[mask_indices] = values - values.mean(axis=0)
        return dtm_field
