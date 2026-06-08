import argparse
import glob
import os

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

DEFAULT_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")
DEFAULT_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")
BLACK_LEVEL = 511
SATURATION_LEVEL = 16383
EPS = 1e-6

SAM_CHECKPOINT_URLS = {
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}


def download_checkpoint(dest_path: str, url: str) -> None:
    from urllib.request import urlretrieve

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"Downloading SAM checkpoint from {url} to {dest_path}...")
    urlretrieve(url, dest_path)
    print("Download complete.")


def load_linear_rgb(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype == np.uint16 or np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32)
        img = np.clip((img - BLACK_LEVEL) / (SATURATION_LEVEL - BLACK_LEVEL), 0.0, 1.0)
    else:
        img = img.astype(np.float32) / 255.0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def resize_for_inference(img: np.ndarray, max_side: int = 1024):
    height, width = img.shape[:2]
    if max(height, width) <= max_side:
        return img, (height, width)
    scale = max_side / max(height, width)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    new_w = max(32, ((new_w + 31) // 32) * 32)
    new_h = max(32, ((new_h + 31) // 32) * 32)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, (new_h, new_w)


def resize_for_sgbm(img: np.ndarray, max_width: int = 300, max_height: int = 200):
    height, width = img.shape[:2]
    if width <= max_width and height <= max_height:
        return img, (height, width)
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, (new_h, new_w)


def merge_masks(masks, height: int, width: int):
    best_score = np.zeros((height, width), dtype=np.float32)
    best_instance = np.full((height, width), -1, dtype=np.int32)
    instance_scores = []

    for idx, mask_data in enumerate(masks):
        mask = mask_data["segmentation"].astype(np.uint8)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        score = float(mask_data.get("predicted_iou", 0.0))
        mask_bool = mask > 0
        update_pixels = mask_bool & (score > best_score)
        if np.any(update_pixels):
            best_score[update_pixels] = score
            best_instance[update_pixels] = idx
        instance_scores.append(score)

    return best_instance, instance_scores


def get_cache_path(output_dir: str, base_name: str) -> str:
    return os.path.join(output_dir, f"{base_name}_sam_cache.npz")


def save_mask_cache(cache_path: str, best_instance: np.ndarray, instance_scores: list[float]) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(cache_path, best_instance=best_instance, instance_scores=np.asarray(instance_scores, dtype=np.float32))


def load_mask_cache(cache_path: str):
    data = np.load(cache_path)
    best_instance = data["best_instance"]
    instance_scores = data["instance_scores"].tolist()
    return best_instance, instance_scores


def get_palette_color(class_id: int):
    if class_id % 10 == 0:
        return (255, 0, 0)
    if class_id % 10 == 1:
        return (0, 255, 0)
    if class_id % 10 == 2:
        return (0, 0, 255)
    if class_id % 10 == 3:
        return (255, 255, 0)
    if class_id % 10 == 4:
        return (255, 0, 255)
    if class_id % 10 == 5:
        return (0, 255, 255)
    if class_id % 10 == 6:
        return (128, 0, 128)
    if class_id % 10 == 7:
        return (0, 128, 128)
    if class_id % 10 == 8:
        return (128, 128, 0)
    return (0, 0, 128)


def build_output_images(img_rgb: np.ndarray, best_instance: np.ndarray):
    img_bgr = cv2.cvtColor(np.clip(np.power(img_rgb, 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    mask_colored = np.zeros_like(img_bgr, dtype=np.uint8)
    unique_ids = np.unique(best_instance)
    for instance_id in unique_ids:
        if instance_id < 0:
            continue
        mask_colored[best_instance == instance_id] = get_palette_color(int(instance_id))
    mask_overlay = cv2.addWeighted(img_bgr, 1.0, mask_colored, 0.5, 0)
    return mask_colored, mask_overlay


def log_chromaticity(img_rgb: np.ndarray) -> np.ndarray:
    img = np.clip(img_rgb, EPS, None)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    log_rg = np.log(r + EPS) - np.log(g + EPS)
    log_gb = np.log(g + EPS) - np.log(b + EPS)
    return np.stack((log_rg, log_gb), axis=-1)


def compute_distance_to_mean(log_chroma: np.ndarray, instance_map: np.ndarray) -> np.ndarray:
    height, width = instance_map.shape
    result = np.zeros((height, width), dtype=np.float32)
    unique_ids = np.unique(instance_map)
    for instance_id in unique_ids:
        if instance_id < 0:
            continue
        mask = instance_map == instance_id
        values = log_chroma[mask]
        if values.size == 0:
            continue
        mean = values.mean(axis=0)
        delta = values - mean
        result[mask] = np.linalg.norm(delta, axis=1)
    return result


def normalize_vector_field(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norm = np.maximum(norm, EPS)
    return vectors / norm


def compute_directional_difference(log_chroma: np.ndarray, instance_map: np.ndarray) -> np.ndarray:
    pixel_dirs = normalize_vector_field(log_chroma)
    height, width = instance_map.shape
    diff = np.zeros((height, width, 2), dtype=np.float32)
    unique_ids = np.unique(instance_map)
    for instance_id in unique_ids:
        if instance_id < 0:
            continue
        mask = instance_map == instance_id
        dirs = pixel_dirs[mask]
        if dirs.size == 0:
            continue
        mean_dir = dirs.mean(axis=0)
        mean_dir_norm = np.linalg.norm(mean_dir)
        if mean_dir_norm < EPS:
            mean_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            mean_dir = mean_dir / mean_dir_norm
        diff[mask] = dirs - mean_dir
    return diff


def compute_directional_dtm(log_chroma_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width, _ = log_chroma_img.shape
    dtm_field = np.zeros_like(log_chroma_img)
    unique_masks = np.unique(mask)
    for m_id in unique_masks:
        if m_id < 0:
            continue
        mask_indices = mask == m_id
        values = log_chroma_img[mask_indices]
        if values.size == 0:
            continue
        mean_color = values.mean(axis=0)
        dtm_field[mask_indices] = values - mean_color
    return dtm_field


def make_dummy_pca():
    class DummyPCA:
        def __init__(self):
            self.components_ = np.zeros((1, 2), dtype=np.float32)
            self.mean_ = np.zeros(2, dtype=np.float32)

        def fit(self, X):
            return self

        def fit_transform(self, X):
            n = X.shape[0]
            return np.zeros((n, 1), dtype=np.float32)

        def transform(self, X):
            n = X.shape[0]
            return np.zeros((n, 1), dtype=np.float32)

        def inverse_transform(self, X):
            X = np.asarray(X, dtype=np.float32)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n = X.shape[0]
            return np.zeros((n, 2), dtype=np.float32)

    return DummyPCA()


def create_label_space(dtm_field: np.ndarray, num_labels: int = 16):
    height, width, _ = dtm_field.shape
    flat_dtm = np.nan_to_num(dtm_field.reshape(-1, 2), nan=0.0, posinf=0.0, neginf=0.0)
    if flat_dtm.size == 0 or np.allclose(flat_dtm, flat_dtm[0], atol=1e-9) or np.allclose(np.var(flat_dtm, axis=0), 0.0, atol=1e-12):
        pca = make_dummy_pca()
        labels_1d = np.zeros(num_labels, dtype=np.float32)
        return labels_1d, pca

    pca = PCA(n_components=1)
    projected = np.nan_to_num(pca.fit_transform(flat_dtm), nan=0.0, posinf=0.0, neginf=0.0)
    min_val, max_val = projected.min(), projected.max()
    if np.isclose(min_val, max_val, atol=1e-12):
        labels_1d = np.zeros(num_labels, dtype=np.float32)
    else:
        labels_1d = np.linspace(min_val, max_val, num_labels)
    return labels_1d, pca


def compute_data_cost(dtm_field: np.ndarray, labels_1d: np.ndarray, pca) -> np.ndarray:
    height, width, _ = dtm_field.shape
    num_labels = len(labels_1d)
    data_cost = np.zeros((height, width, num_labels), dtype=np.float32)
    flat_dtm = np.nan_to_num(dtm_field.reshape(-1, 2), nan=0.0, posinf=0.0, neginf=0.0)
    projected_dtm = np.nan_to_num(pca.transform(flat_dtm), nan=0.0, posinf=0.0, neginf=0.0).reshape(height, width)
    for l_idx, label in enumerate(labels_1d):
        data_cost[:, :, l_idx] = (projected_dtm - label) ** 2
    return data_cost


def compute_path_cost(data_cost: np.ndarray, mask: np.ndarray, direction: tuple[int, int], P1_base: float, P2_base: float) -> np.ndarray:
    height, width, num_labels = data_cost.shape
    dx, dy = direction
    L = np.zeros_like(data_cost, dtype=np.float32)
    L[:] = data_cost[:]
    x_range = range(1, height) if dx >= 0 else range(height - 2, -1, -1)
    y_range = range(1, width) if dy >= 0 else range(width - 2, -1, -1)
    for x in x_range:
        for y in y_range:
            px, py = x - dx, y - dy
            if 0 <= px < height and 0 <= py < width:
                if mask[x, y] == mask[px, py]:
                    P2 = P2_base
                else:
                    P2 = P2_base * 0.1
                prev_costs = L[px, py, :]
                for l in range(num_labels):
                    cost_same = prev_costs[l]
                    cost_step_up = prev_costs[l - 1] + P1_base if l > 0 else float('inf')
                    cost_step_down = prev_costs[l + 1] + P1_base if l < num_labels - 1 else float('inf')
                    cost_jump = prev_costs.min() + P2
                    L[x, y, l] = data_cost[x, y, l] + min(cost_same, cost_step_up, cost_step_down, cost_jump) - prev_costs.min()
    return L


def semi_global_matching(data_cost: np.ndarray, mask: np.ndarray, direction_count: int = 4, P1: float = 0.05, P2: float = 0.8) -> np.ndarray:
    if direction_count == 4:
        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ]
    else:
        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ]
    S = np.zeros_like(data_cost, dtype=np.float32)
    total = len(directions)
    for idx, direction in enumerate(directions):
        percent = int((idx + 1) / total * 100)
        print(f"[SGBM] Path aggregation {idx + 1}/{total} ({percent}%)")
        S += compute_path_cost(data_cost, mask, direction, P1, P2)
    return S


def winner_take_all(aggregated_cost: np.ndarray, labels_1d: np.ndarray, pca) -> np.ndarray:
    best_label_indices = np.argmin(aggregated_cost, axis=2)
    illuminant_map_1d = labels_1d[best_label_indices]
    height, width = illuminant_map_1d.shape
    illuminant_map_2d = pca.inverse_transform(illuminant_map_1d.reshape(-1, 1))
    return illuminant_map_2d.reshape(height, width, 2)


def visualize_2d_map_as_hsv(map_2d: np.ndarray) -> np.ndarray:
    clipped = np.nan_to_num(map_2d)
    magnitudes = np.linalg.norm(clipped, axis=-1)
    angles = np.arctan2(clipped[..., 1], clipped[..., 0])
    hue = ((angles / (2 * np.pi)) + 1.0) % 1.0
    saturation = np.ones_like(hue, dtype=np.float32)
    value = (magnitudes - magnitudes.min()) / max(magnitudes.max() - magnitudes.min(), EPS)
    hsv = np.stack((hue, saturation, value), axis=-1).astype(np.float32)
    hsv[:, :, 0] *= 179.0
    hsv[:, :, 1:] *= 255.0
    hsv_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return hsv_bgr


def run_sgbm_color_constancy(log_chroma_img: np.ndarray, mask: np.ndarray, direction_count: int = 4, num_labels: int = 16):
    print("[SGBM] Computing directional DTM field...")
    dtm_field = compute_directional_dtm(log_chroma_img, mask)
    print("[SGBM] Creating label space (PCA)...")
    labels_1d, pca = create_label_space(dtm_field, num_labels=num_labels)
    print("[SGBM] Computing data cost...")
    data_cost = compute_data_cost(dtm_field, labels_1d, pca)
    print(f"[SGBM] Running semi-global matching in {direction_count} directions...")
    aggregated_cost = semi_global_matching(data_cost, mask, direction_count=direction_count, P1=0.02, P2=0.5)
    print("[SGBM] Applying winner-take-all to get illuminant map...")
    best_label_indices = np.argmin(aggregated_cost, axis=2)
    illuminant_map = winner_take_all(aggregated_cost, labels_1d, pca)
    print("[SGBM] Color-constancy map complete.")
    return illuminant_map, best_label_indices, labels_1d, pca


def direction_to_color(direction: np.ndarray) -> tuple[int, int, int]:
    angle = np.arctan2(direction[1], direction[0])
    hue = (angle / (2 * np.pi) + 1.0) % 1.0
    saturation = 1.0
    value = 1.0
    h_i = int(hue * 6)
    f = hue * 6 - h_i
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)
    if h_i == 0:
        r, g, b = value, t, p
    elif h_i == 1:
        r, g, b = q, value, p
    elif h_i == 2:
        r, g, b = p, value, t
    elif h_i == 3:
        r, g, b = p, q, value
    elif h_i == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_direction_field(image_bgr: np.ndarray, instance_map: np.ndarray, vector_diff: np.ndarray, pixel_dirs: np.ndarray, subsample: int = 16, scale: float = 8.0) -> np.ndarray:
    overlay = image_bgr.copy()
    height, width = instance_map.shape
    for y in range(subsample // 2, height, subsample):
        for x in range(subsample // 2, width, subsample):
            if instance_map[y, x] < 0:
                continue
            dx, dy = vector_diff[y, x]
            color = direction_to_color(pixel_dirs[y, x])
            start = (int(x), int(y))
            end = (int(x + dx * scale * subsample), int(y + dy * scale * subsample))
            cv2.arrowedLine(overlay, start, end, color, 1, tipLength=0.3)
    return overlay


def mask_bounding_box(mask: np.ndarray, pad: int = 4):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = max(0, ys.min() - pad)
    y1 = min(mask.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(mask.shape[1], xs.max() + pad + 1)
    return y0, y1, x0, x1


def draw_cropped_mask_vectors(image_bgr: np.ndarray, instance_map: np.ndarray, vector_diff: np.ndarray, pixel_dirs: np.ndarray, instance_id: int, subsample: int = 4, scale: float = 4.0) -> np.ndarray:
    mask = instance_map == instance_id
    bbox = mask_bounding_box(mask)
    if bbox is None:
        return None
    y0, y1, x0, x1 = bbox
    crop = image_bgr[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop_diff = vector_diff[y0:y1, x0:x1]
    crop_dirs = pixel_dirs[y0:y1, x0:x1]
    h, w = crop_mask.shape
    for cy in range(subsample // 2, h, subsample):
        for cx in range(subsample // 2, w, subsample):
            if not crop_mask[cy, cx]:
                continue
            dx, dy = crop_diff[cy, cx]
            color = direction_to_color(crop_dirs[cy, cx])
            start = (int(cx), int(cy))
            end = (int(cx + dx * scale * subsample), int(cy + dy * scale * subsample))
            cv2.arrowedLine(crop, start, end, color, 1, tipLength=0.3)
    return crop


def gamma_encode_linear(img_linear: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    img = np.clip(img_linear, 0.0, 1.0)
    img_gamma = np.power(img, 1.0 / gamma)
    return np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))
    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - min_val) / (max_val - min_val)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def apply_sgbm_white_balance(linear_img: np.ndarray, illuminant_map: np.ndarray) -> np.ndarray:
    if illuminant_map.shape[:2] != linear_img.shape[:2]:
        illuminant_map = cv2.resize(illuminant_map, (linear_img.shape[1], linear_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    log_rg_illum = illuminant_map[..., 0]
    log_gb_illum = illuminant_map[..., 1]
    gain_r = np.exp(-log_rg_illum)
    gain_g = np.ones_like(gain_r, dtype=np.float32)
    gain_b = np.exp(log_gb_illum)
    gain_geo = np.cbrt(gain_r * gain_g * gain_b)
    gain_r /= gain_geo
    gain_g /= gain_geo
    gain_b /= gain_geo

    corrected = np.empty_like(linear_img, dtype=np.float32)
    corrected[..., 0] = linear_img[..., 0] * gain_r
    corrected[..., 1] = linear_img[..., 1] * gain_g
    corrected[..., 2] = linear_img[..., 2] * gain_b
    corrected = np.clip(corrected, 0.0, 1.0)
    return corrected


def normalize_distance_by_mask(distance_map: np.ndarray, instance_map: np.ndarray) -> np.ndarray:
    height, width = distance_map.shape
    normalized = np.zeros((height, width), dtype=np.uint8)
    unique_ids = np.unique(instance_map)
    for instance_id in unique_ids:
        if instance_id < 0:
            continue
        mask = instance_map == instance_id
        values = distance_map[mask]
        if values.size == 0:
            continue
        min_val = float(values.min())
        max_val = float(values.max())
        if max_val <= min_val:
            normalized[mask] = 0
        else:
            scaled = ((values - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            normalized[mask] = scaled
    return normalized


def collect_tiff_paths(root_dir: str) -> list[str]:
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    paths = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(os.path.join(root_dir, pattern))))
    return paths


def process_tiff(image_path: str, generator, output_dir: str, max_side: int, sgbm_directions: int = 4, sgbm_max_width: int = 300, sgbm_max_height: int = 200):
    print(f"Processing TIFF: {image_path}")
    img_rgb = load_linear_rgb(image_path)
    height, width = img_rgb.shape[:2]
    base = os.path.splitext(os.path.basename(image_path))[0]
    cache_path = get_cache_path(output_dir, base)

    best_instance = None
    instance_scores = None
    needs_generation = True

    if os.path.exists(cache_path):
        print(f"Loading cached SAM masks for {image_path}")
        cached_instance, cached_scores = load_mask_cache(cache_path)
        if cached_instance.shape == (height, width):
            best_instance = cached_instance
            instance_scores = cached_scores
            needs_generation = False
            print(f"Using cached SAM instance map from {cache_path}")
        else:
            print(f"Cached mask shape {cached_instance.shape} does not match image shape {(height, width)}. Regenerating SAM masks.")

    if needs_generation:
        infer_img, _ = resize_for_inference(img_rgb, max_side=max_side)

        masks = generator.generate(infer_img)
        if len(masks) == 0:
            print(f"No masks generated for {image_path}")
            return

        if infer_img.shape[:2] != (height, width):
            for mask_data in masks:
                seg = mask_data["segmentation"].astype(np.uint8)
                mask_data["segmentation"] = cv2.resize(seg, (width, height), interpolation=cv2.INTER_NEAREST)

        best_instance, instance_scores = merge_masks(masks, height, width)
        save_mask_cache(cache_path, best_instance, instance_scores)
        print(f"Saved SAM cache to {cache_path}")
    distance_map = compute_distance_to_mean(log_chromaticity(img_rgb), best_instance)

    base = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    dist_npy_path = os.path.join(output_dir, f"{base}_distance_to_mean.npy")
    dist_png_path = os.path.join(output_dir, f"{base}_distance_to_mean.png")
    dirs_npy_path = os.path.join(output_dir, f"{base}_directional_diff.npy")
    field_png_path = os.path.join(output_dir, f"{base}_directional_field.png")
    mask_colored_path = os.path.join(output_dir, f"{base}_mask_colored.png")
    mask_overlay_path = os.path.join(output_dir, f"{base}_mask_overlay.png")
    classes_path = os.path.join(output_dir, f"{base}_mask_classes.txt")

    np.save(dist_npy_path, distance_map)
    cv2.imwrite(dist_png_path, normalize_distance_by_mask(distance_map, best_instance))

    log_chroma = log_chromaticity(img_rgb)
    pixel_dirs = normalize_vector_field(log_chroma)
    directional_diff = compute_directional_difference(log_chroma, best_instance)
    np.save(dirs_npy_path, directional_diff)

    sgbm_img, (sgbm_h, sgbm_w) = resize_for_sgbm(img_rgb, max_width=sgbm_max_width, max_height=sgbm_max_height)
    log_chroma_small = log_chromaticity(sgbm_img)
    resized_instance = cv2.resize(best_instance.astype(np.int32), (sgbm_w, sgbm_h), interpolation=cv2.INTER_NEAREST)
    sgbm_illuminant_map_small, sgbm_label_indices, sgbm_labels_1d, sgbm_pca = run_sgbm_color_constancy(
        log_chroma_small,
        resized_instance,
        direction_count=sgbm_directions,
    )
    sgbm_npy_path = os.path.join(output_dir, f"{base}_sgbm_illuminant_map_small.npy")
    sgbm_vis_path = os.path.join(output_dir, f"{base}_sgbm_illuminant_map_small_vis.png")
    labels_path = os.path.join(output_dir, f"{base}_sgbm_label_indices.npy")
    labels_1d_path = os.path.join(output_dir, f"{base}_sgbm_label_values.npy")
    np.save(sgbm_npy_path, sgbm_illuminant_map_small)
    np.save(labels_path, sgbm_label_indices)
    np.save(labels_1d_path, sgbm_labels_1d)
    vis_map = visualize_2d_map_as_hsv(sgbm_illuminant_map_small)
    cv2.imwrite(sgbm_vis_path, vis_map)

    wb_linear = apply_sgbm_white_balance(img_rgb, sgbm_illuminant_map_small)
    wb_npy_path = os.path.join(output_dir, f"{base}_wb_linear.npy")
    wb_gamma_path = os.path.join(output_dir, f"{base}_wb_gamma.png")
    np.save(wb_npy_path, wb_linear)
    wb_gamma_bgr = cv2.cvtColor(gamma_encode_linear(wb_linear), cv2.COLOR_RGB2BGR)
    cv2.imwrite(wb_gamma_path, wb_gamma_bgr)

    img_bgr = cv2.cvtColor(np.clip(np.power(img_rgb, 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    field_overlay = draw_direction_field(img_bgr, best_instance, directional_diff, pixel_dirs, subsample=16, scale=4.0)
    cv2.imwrite(field_png_path, field_overlay)

    masks_dir = os.path.join(output_dir, base, "mask_vectors")
    os.makedirs(masks_dir, exist_ok=True)
    for instance_id in np.unique(best_instance):
        if instance_id < 0:
            continue
        crop_vis = draw_cropped_mask_vectors(img_bgr, best_instance, directional_diff, pixel_dirs, instance_id, subsample=4, scale=4.0)
        if crop_vis is None:
            continue
        crop_path = os.path.join(masks_dir, f"mask_{instance_id:03d}_vectors.png")
        cv2.imwrite(crop_path, crop_vis)

    mask_colored, mask_overlay = build_output_images(img_rgb, best_instance)
    cv2.imwrite(mask_colored_path, mask_colored)
    cv2.imwrite(mask_overlay_path, mask_overlay)

    with open(classes_path, "w", encoding="utf-8") as f:
        for instance_id, score in enumerate(instance_scores):
            color = get_palette_color(instance_id)
            f.write(f"mask_{instance_id} score={score:.3f} rgb({color[2]},{color[1]},{color[0]})\n")

    print(f"Saved distance map: {dist_png_path} and {dist_npy_path}")
    print(f"Saved directional diff: {dirs_npy_path} and {field_png_path}")
    print(f"Saved SAM segmentation outputs: {mask_colored_path}, {mask_overlay_path}, {classes_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute per-pixel distance to mean 2D log chromaticity for SAM2 panoptic masks on TIFF images.")
    parser.add_argument("--input-dir", type=str, default="linear_tiff_selected_ones", help="Directory containing TIFF images.")
    parser.add_argument("--output-dir", type=str, default="distance_to_mean_outputs", help="Directory to save distance maps and SAM outputs.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="SAM checkpoint path.")
    parser.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE, choices=["default", "vit_h", "vit_l", "vit_b"], help="SAM model type.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for SAM inference.")
    parser.add_argument("--points-per-side", type=int, default=16, help="Grid density for automatic mask generation.")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.2, help="Mask IOU threshold for automatic generation.")
    parser.add_argument("--sgbm-directions", type=int, choices=[4, 8], default=4, help="Number of directions for SGBM aggregation.")
    parser.add_argument("--sgbm-max-width", type=int, default=300, help="Maximum width for resized SGBM processing.")
    parser.add_argument("--sgbm-max-height", type=int, default=200, help="Maximum height for resized SGBM processing.")
    parser.add_argument("--stability-score-thresh", type=float, default=0.2, help="Mask stability score threshold.")
    parser.add_argument("--crop-n-layers", type=int, default=0, help="Number of crop layers for automatic mask generation.")
    parser.add_argument("--min-mask-region-area", type=int, default=1000, help="Minimum area for masks to keep.")
    parser.add_argument("--max-side", type=int, default=1024, help="Maximum image side length for SAM inference.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        basename = os.path.basename(args.checkpoint)
        if basename in SAM_CHECKPOINT_URLS:
            download_checkpoint(args.checkpoint, SAM_CHECKPOINT_URLS[basename])
        else:
            raise FileNotFoundError(f"SAM checkpoint not found: {args.checkpoint}")

    print(f"Loading SAM model ({args.model_type}) from {args.checkpoint}...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    if args.device.startswith("cuda"):
        sam.to(dtype=torch.float16)

    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_region_area,
    )

    tiff_paths = collect_tiff_paths(args.input_dir)
    if not tiff_paths:
        print(f"No TIFF files found in {args.input_dir}")
        return

    for image_path in tiff_paths:
        process_tiff(
            image_path,
            generator,
            args.output_dir,
            args.max_side,
            args.sgbm_directions,
            args.sgbm_max_width,
            args.sgbm_max_height,
        )


if __name__ == "__main__":
    main()
