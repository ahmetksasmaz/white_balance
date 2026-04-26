import json
import datetime
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import cv2 as cv
import numpy as np
from tqdm import tqdm

from datasets.data import Data
from datasets.cubepp.cubepp_dataprovider import CubePPDataProvider
from datasets.lsmi.lsmi_dataprovider import LSMIDataProvider
from datasets.gehler.gehler_dataprovider import GehlerDataProvider
from datasets.nus8.nus8_dataprovider import NUS8DataProvider
from datasets.nus8.nus8_extended_dataprovider import NUS8ExtendedDataProvider
from datasets.miniature.miniature_dataprovider import MiniatureDataProvider
from datasets.reallife.reallife_dataprovider import ReallifeDataProvider

from white_balance_algorithms.gray_world.gray_world_naive import GrayWorldNaive
from white_balance_algorithms.gray_world.gray_world_95_boundaries_all_channels import GrayWorld95BoundariesAllChannels
from white_balance_algorithms.gray_world.gray_world_95_boundaries_any_channel import GrayWorld95BoundariesAnyChannel
from white_balance_algorithms.svwb_unet.svwb_unet_default import SVWBUnet

from white_balance_algorithms.max_rgb.max_rgb_naive import MaxRGBNaive
from white_balance_algorithms.max_rgb.max_rgb_99_percentile import MaxRGB99Percentile
from white_balance_algorithms.max_rgb.max_rgb_95_percentile import MaxRGB95Percentile
from white_balance_algorithms.max_rgb.max_rgb_gaussian import MaxRGBGaussian
from white_balance_algorithms.max_rgb.max_rgb_gaussian_99_percentile import MaxRGBGaussian99Percentile
from white_balance_algorithms.max_rgb.max_rgb_gaussian_95_percentile import MaxRGBGaussian95Percentile
from white_balance_algorithms.max_rgb.max_rgb_median import MaxRGBMedian
from white_balance_algorithms.max_rgb.max_rgb_median_99_percentile import MaxRGBMedian99Percentile
from white_balance_algorithms.max_rgb.max_rgb_median_95_percentile import MaxRGBMedian95Percentile

from white_balance_algorithms.shades_of_gray.shades_of_gray_default import ShadesOfGrayDefault
from white_balance_algorithms.shades_of_gray.shades_of_gray_p3 import ShadesOfGrayP3
from white_balance_algorithms.shades_of_gray.shades_of_gray_p4 import ShadesOfGrayP4
from white_balance_algorithms.shades_of_gray.shades_of_gray_masked_default import ShadesOfGrayMaskedDefault
from white_balance_algorithms.shades_of_gray.shades_of_gray_masked_p3 import ShadesOfGrayMaskedP3

from white_balance_algorithms.fast_awb.fast_awb_default import FastAWBDefault
from white_balance_algorithms.fast_awb.fast_awb_p6 import FastAWBP6

from white_balance_algorithms.cheng.cheng_prc_0_5 import ChengPrc05
from white_balance_algorithms.cheng.cheng_prc_3 import ChengPrc3

from visuals.log_chrominance_histogram import LogChrominanceHistogram
from visuals.normalized_rgb_histogram import NormalizedRGBHistogram


DATASET_PROVIDERS = {
    "cubepp": CubePPDataProvider,
    "lsmi": LSMIDataProvider,
    "gehler": GehlerDataProvider,
    "nus8": NUS8DataProvider,
    "nus8extended": NUS8ExtendedDataProvider,
    "miniature": MiniatureDataProvider,
    "reallife": ReallifeDataProvider,
}

ALGORITHM_REGISTRY = {
    ("gray_world", "naive"): GrayWorldNaive,
    ("gray_world", "95_boundaries_all_channels"): GrayWorld95BoundariesAllChannels,
    ("gray_world", "95_boundaries_any_channel"): GrayWorld95BoundariesAnyChannel,
    ("max_rgb", "naive"): MaxRGBNaive,
    ("max_rgb", "99_percentile"): MaxRGB99Percentile,
    ("max_rgb", "95_percentile"): MaxRGB95Percentile,
    ("max_rgb", "gaussian"): MaxRGBGaussian,
    ("max_rgb", "gaussian_99_percentile"): MaxRGBGaussian99Percentile,
    ("max_rgb", "gaussian_95_percentile"): MaxRGBGaussian95Percentile,
    ("max_rgb", "median"): MaxRGBMedian,
    ("max_rgb", "median_99_percentile"): MaxRGBMedian99Percentile,
    ("max_rgb", "median_95_percentile"): MaxRGBMedian95Percentile,
    ("shades_of_gray", "default"): ShadesOfGrayDefault,
    ("shades_of_gray", "p3"): ShadesOfGrayP3,
    ("shades_of_gray", "p4"): ShadesOfGrayP4,
    ("shades_of_gray", "masked_default"): ShadesOfGrayMaskedDefault,
    ("shades_of_gray", "masked_p3"): ShadesOfGrayMaskedP3,
    ("fast_awb", "default"): FastAWBDefault,
    ("fast_awb", "p6"): FastAWBP6,
    ("cheng", "prc_0_5"): ChengPrc05,
    ("cheng", "prc_3"): ChengPrc3,
    ("svwb_unet", "default"): SVWBUnet,
}


def _extract_camera(dataset_name, data_provider, index):
    """Extract camera identifier from the data provider's internal path info."""
    if dataset_name == "cubepp":
        return "default_camera"

    image_path = data_provider.data_names[index]

    if dataset_name == "gehler":
        if "canon1d" in image_path:
            return "canon1d"
        elif "canon5d" in image_path:
            return "canon5d"
        return "unknown"

    if dataset_name in ("nus8", "nus8extended"):
        # Path pattern: .../CameraName/PNG/image.PNG
        parts = image_path.replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part == "PNG" and i > 0:
                return parts[i - 1]
        return "unknown"

    if dataset_name == "lsmi":
        # Path pattern: .../camera_model/place/file
        path_lower = image_path.lower()
        if "/galaxy/" in path_lower:
            return "galaxy"
        elif "/nikon/" in path_lower:
            return "nikon"
        elif "/sony/" in path_lower:
            return "sony"
        return "unknown"

    return "unknown"


def _worker_fn(task_info):
    """Worker function for multi-processing."""
    dataset_name, algo_name, variant_name, idx, process_masked, camera_filter, saturation_mask_str, color_checker_str, export_corrected_images, export_resize_factor, output_path = task_info
    try:
        saturation_masks = {
            'none': None,
            'raw_all_98': ('raw', 'all', 0.98),
            'raw_all_100': ('raw', 'all', 1.0),
            'raw_any_98': ('raw', 'any', 0.98),
            'raw_any_100': ('raw', 'any', 1.0),
            'normalized_all_98': ('normalized', 'all', 0.98),
            'normalized_all_100': ('normalized', 'all', 1.0),
            'normalized_any_98': ('normalized', 'any', 0.98),
            'normalized_any_100': ('normalized', 'any', 1.0),
        }
        saturation_mask_tuple = saturation_masks[saturation_mask_str]

        # Instantiate inside worker to avoid pickle issues with some objects
        if dataset_name in ("nus8", "nus8extended", "gehler"):
            data_provider = DATASET_PROVIDERS[dataset_name](saturation_mask=saturation_mask_tuple, color_checker=color_checker_str)
        else:
            data_provider = DATASET_PROVIDERS[dataset_name]()
        algorithm = ALGORITHM_REGISTRY[(algo_name, variant_name)]()

        camera = _extract_camera(dataset_name, data_provider, idx)

        data = data_provider[idx]
        data.set_camera(camera)
        image_name = data.get_image_name()

        estimations = algorithm.estimate(data, process_masked=process_masked)
        error_metrics = data.compute_error_metrics(estimations)

        all_data = data.get_data()

        exported_paths = {
            "masked_grid_path": None,
            "illuminant_map_path": None,
        }
        if export_corrected_images:
            export_dir = _prepare_export_paths(output_path, dataset_name, algo_name, variant_name, image_name)
            corrected_raw = _get_corrected_raw_image_from_estimations(data.get_raw_image(), estimations)
            masked_grid = _get_masked_grid_image(
                data,
                corrected_raw,
                estimations.get("single_illuminant"),
                image_size=300,
                apply_mask=process_masked,
            )
            if masked_grid is not None:
                filename = f"{dataset_name}_{image_name}_{algo_name}_{variant_name}_masked_grid.png"
                path = os.path.join(export_dir, filename)
                if _export_image_array(masked_grid, path, export_resize_factor):
                    exported_paths["masked_grid_path"] = path
            illuminant_map = estimations.get("illuminant_map")
            if illuminant_map is not None:
                filename = f"{dataset_name}_{image_name}_{algo_name}_{variant_name}_illuminant_map.png"
                path = os.path.join(export_dir, filename)
                if _export_illuminant_map_as_image(illuminant_map, path, data.get_mask() if process_masked else None):
                    exported_paths["illuminant_map_path"] = path

        single_illuminant_value = estimations.get("single_illuminant")
        if single_illuminant_value is not None:
            single_illuminant_value = [str(single_illuminant_value[0]), str(single_illuminant_value[1])]

        return {
            "dataset": dataset_name,
            "camera": camera,
            "image_name": image_name,
            "algorithm": algo_name,
            "variant": variant_name,
            "estimations": {
                "single_illuminant": single_illuminant_value,
                "masked_grid_path": exported_paths["masked_grid_path"],
                "illuminant_map_path": exported_paths["illuminant_map_path"],
            },
            "ground_truths": {
                "illuminants": _serialize_error_metrics(all_data["illuminants"]),
                #"illuminant_map": all_data["illuminant_map"],
                #"srgb_image": all_data["srgb_image"]
            },
            "errors": _serialize_error_metrics(error_metrics),
        }
    except Exception as e:
        return {
            "dataset": dataset_name,
            "camera": "unknown",
            "image_name": f"index_{idx}",
            "algorithm": algo_name,
            "variant": variant_name,
            "estimations": None,
            "ground_truths": None,
            "errors": None,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }


def _serialize_error_metrics(error_metrics):
    """Convert error metrics dict to a JSON-safe dict."""
    result = {}
    for key, value in error_metrics.items():
        if value is None:
            result[key] = None
        elif isinstance(value, dict):
            result[key] = {}
            for k, v in value.items():
                if hasattr(v, 'item'):  # numpy scalar
                    result[key][k] = v.item()
                elif isinstance(v, float):
                    result[key][k] = v
                else:
                    result[key][k] = v
        else:
            result[key] = value
    return result


def _prepare_display_image(image):
    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.1:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / 2.2)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img


def _get_masked_display_image(data):
    raw_img = data.get_raw_image()
    srgb_img = data.get_srgb_image()
    mask = data.get_mask()
    if mask is None or (raw_img is None and srgb_img is None):
        return None

    display_source = raw_img if raw_img is not None else srgb_img
    display_img = _prepare_display_image(display_source)

    mask_arr = mask.astype(bool) if isinstance(mask, np.ndarray) else np.array(mask, dtype=bool)
    if mask_arr.shape != display_img.shape[:2]:
        try:
            mask_arr = cv.resize(mask_arr.astype(np.uint8), (display_img.shape[1], display_img.shape[0]), interpolation=cv.INTER_NEAREST).astype(bool)
        except Exception:
            mask_arr = np.ones(display_img.shape[:2], dtype=bool)

    masked_display = display_img.copy()
    masked_display[~mask_arr] = 0
    return masked_display


def _apply_von_kries_single(raw_image, single_illuminant):
    if raw_image is None or single_illuminant is None:
        return None
    try:
        r_g, b_g = single_illuminant
        r_scale = 1.0 / float(r_g) if float(r_g) != 0 else 0.0
        b_scale = 1.0 / float(b_g) if float(b_g) != 0 else 0.0
        scale = np.array([b_scale, 1.0, r_scale], dtype=np.float32)
        corrected = raw_image.astype(np.float32) * scale.reshape((1, 1, 3))
        return np.clip(corrected, 0.0, 1.0)
    except Exception:
        return None


def _apply_von_kries_map(raw_image, illuminant_map):
    if raw_image is None or illuminant_map is None:
        return None

    if illuminant_map.ndim == 3:
        if illuminant_map.shape[2] == 2:
            r_g = illuminant_map[..., 0].astype(np.float32)
            b_g = illuminant_map[..., 1].astype(np.float32)
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_r = np.where(r_g != 0, 1.0 / r_g, 0.0)
                inv_b = np.where(b_g != 0, 1.0 / b_g, 0.0)
            inv_map = np.stack([inv_b, np.ones_like(inv_b), inv_r], axis=-1)
            corrected = raw_image.astype(np.float32) * inv_map
            return np.clip(corrected, 0.0, 1.0)
        elif illuminant_map.shape[2] == 3:
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_map = np.where(illuminant_map != 0, 1.0 / illuminant_map, 0.0)
            corrected = raw_image.astype(np.float32) * inv_map
            return np.clip(corrected, 0.0, 1.0)
    return None


def _get_corrected_raw_image_from_estimations(raw_image, estimations):
    if estimations is None:
        return None
    corrected_raw = estimations.get("multi_illuminant_corrected_raw_image")
    if corrected_raw is not None:
        return corrected_raw
    corrected_raw = estimations.get("single_illuminant_corrected_raw_image")
    if corrected_raw is not None:
        return corrected_raw
    illuminant_map = estimations.get("illuminant_map")
    if illuminant_map is not None:
        return _apply_von_kries_map(raw_image, illuminant_map)
    return None


def _get_first_ground_truth_illuminant(data):
    illuminants = data.get_illuminants()
    if illuminants is None:
        return None
    for value in illuminants.values():
        if value is not None:
            return value
    return None


def _draw_labeled_text(image, text, position, font_scale=0.65, font_thickness=2, text_color=(255, 255, 255), outline_color=(0, 0, 0)):
    if image is None or text is None:
        return image
    labeled = image.copy()
    font = cv.FONT_HERSHEY_COMPLEX
    cv.putText(labeled, text, position, font, font_scale, outline_color, font_thickness + 1, cv.LINE_AA)
    cv.putText(labeled, text, position, font, font_scale, text_color, font_thickness, cv.LINE_AA)
    return labeled


def _draw_illuminant_label(image, illuminant, label="GT"):
    if image is None or illuminant is None:
        return image
    text = f"{label}: {illuminant[0]:0.2f}, {illuminant[1]:0.2f}"
    return _draw_labeled_text(image, text, (10, 25), font_scale=0.7, font_thickness=2)


def _get_masked_grid_image(data, corrected_raw, estimated_illuminant, image_size=300, apply_mask=True):
    raw_img = data.get_raw_image()
    if raw_img is None:
        return None

    mask = data.get_mask() if apply_mask else None
    mask_arr = None
    if mask is not None:
        mask_arr = mask.astype(bool) if isinstance(mask, np.ndarray) else np.array(mask, dtype=bool)
        if mask_arr.shape != raw_img.shape[:2]:
            try:
                mask_arr = cv.resize(mask_arr.astype(np.uint8), (raw_img.shape[1], raw_img.shape[0]), interpolation=cv.INTER_NEAREST).astype(bool)
            except Exception:
                return None

    masked_raw = raw_img.copy()
    if mask_arr is not None:
        masked_raw[~mask_arr] = 0

    if corrected_raw is None:
        corrected_raw = masked_raw.copy()
    else:
        corrected_raw = corrected_raw.astype(np.float32)
        if corrected_raw.shape[:2] != raw_img.shape[:2]:
            corrected_raw = cv.resize(corrected_raw, (raw_img.shape[1], raw_img.shape[0]), interpolation=cv.INTER_LINEAR)
        corrected_raw = corrected_raw.copy()
        if mask_arr is not None:
            corrected_raw[~mask_arr] = 0

    gt_illuminant = _get_first_ground_truth_illuminant(data)
    gt_corrected_raw = None
    if gt_illuminant is not None:
        gt_corrected_raw = _apply_von_kries_single(masked_raw, gt_illuminant)
        if gt_corrected_raw is not None:
            gt_corrected_raw = gt_corrected_raw.copy()
            if mask_arr is not None:
                gt_corrected_raw[~mask_arr] = 0

    input_data = Data()
    input_data.set_image_name(data.get_image_name())
    input_data.set_raw_image(masked_raw)
    if mask_arr is not None:
        input_data.set_mask(mask_arr)

    corrected_data = Data()
    corrected_data.set_image_name(data.get_image_name())
    corrected_data.set_raw_image(corrected_raw)
    if mask_arr is not None:
        corrected_data.set_mask(mask_arr)

    gt_data = None
    if gt_corrected_raw is not None:
        gt_data = Data()
        gt_data.set_image_name(data.get_image_name())
        gt_data.set_raw_image(gt_corrected_raw)
        gt_data.set_mask(mask_arr)

    log_vis = LogChrominanceHistogram(image_size=image_size)
    rgb_vis = NormalizedRGBHistogram(image_size=image_size)
    input_log, _ = log_vis.visualize(input_data)
    corrected_log, _ = log_vis.visualize(corrected_data)
    gt_log = None
    input_rgb, _ = rgb_vis.visualize(input_data)
    corrected_rgb, _ = rgb_vis.visualize(corrected_data)
    gt_rgb = None

    if gt_data is not None:
        gt_log, _ = log_vis.visualize(gt_data)
        gt_rgb, _ = rgb_vis.visualize(gt_data)

    if input_log is None or corrected_log is None or input_rgb is None or corrected_rgb is None:
        return None

    def _blank_cell():
        return np.full((image_size, image_size, 3), 32, dtype=np.uint8)

    if gt_log is None:
        gt_log = _blank_cell()
    if gt_rgb is None:
        gt_rgb = _blank_cell()

    input_display = _prepare_display_image(masked_raw)
    corrected_display = _prepare_display_image(corrected_raw)
    gt_display = _blank_cell() if gt_corrected_raw is None else _prepare_display_image(gt_corrected_raw)

    input_display = cv.resize(input_display, (image_size, image_size), interpolation=cv.INTER_AREA)
    corrected_display = cv.resize(corrected_display, (image_size, image_size), interpolation=cv.INTER_AREA)
    gt_display = cv.resize(gt_display, (image_size, image_size), interpolation=cv.INTER_AREA)

    input_display = _draw_labeled_text(input_display, "Input", (10, 25), font_scale=0.75, font_thickness=2)
    corrected_display = _draw_labeled_text(corrected_display, "Corrected with Est.", (10, 25), font_scale=0.7, font_thickness=2)
    if gt_corrected_raw is not None:
        gt_display = _draw_labeled_text(gt_display, "Corrected with GT", (10, 25), font_scale=0.7, font_thickness=2)

    if estimated_illuminant is not None:
        estimated_text = f"({estimated_illuminant[0]:0.2f},{estimated_illuminant[1]:0.2f})"
        corrected_display = _draw_labeled_text(corrected_display, estimated_text, (10, image_size - 12), font_scale=0.6, font_thickness=2)

    if gt_illuminant is not None and gt_corrected_raw is not None:
        gt_text = f"({gt_illuminant[0]:0.2f},{gt_illuminant[1]:0.2f})"
        gt_display = _draw_labeled_text(gt_display, gt_text, (10, image_size - 12), font_scale=0.6, font_thickness=2)

    if input_log.shape[:2] != (image_size, image_size):
        input_log = cv.resize(input_log, (image_size, image_size), interpolation=cv.INTER_AREA)
    if corrected_log.shape[:2] != (image_size, image_size):
        corrected_log = cv.resize(corrected_log, (image_size, image_size), interpolation=cv.INTER_AREA)
    if gt_log.shape[:2] != (image_size, image_size):
        gt_log = cv.resize(gt_log, (image_size, image_size), interpolation=cv.INTER_AREA)

    if input_rgb.shape[:2] != (image_size, image_size):
        input_rgb = cv.resize(input_rgb, (image_size, image_size), interpolation=cv.INTER_AREA)
    if corrected_rgb.shape[:2] != (image_size, image_size):
        corrected_rgb = cv.resize(corrected_rgb, (image_size, image_size), interpolation=cv.INTER_AREA)
    if gt_rgb.shape[:2] != (image_size, image_size):
        gt_rgb = cv.resize(gt_rgb, (image_size, image_size), interpolation=cv.INTER_AREA)

    separator_color = (64, 64, 64)
    spacer = np.full((image_size, 10, 3), separator_color, dtype=np.uint8)
    row1 = np.concatenate([input_display, spacer, corrected_display, spacer, gt_display], axis=1)
    row2 = np.concatenate([input_log, spacer, corrected_log, spacer, gt_log], axis=1)
    row3 = np.concatenate([input_rgb, spacer, corrected_rgb, spacer, gt_rgb], axis=1)

    row_spacer = np.full((10, row1.shape[1], 3), separator_color, dtype=np.uint8)
    return np.concatenate([row1, row_spacer, row2, row_spacer, row3], axis=0)


def _export_image_array(image, image_path, resize_factor=None):
    if image is None:
        return False

    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)

    if img.dtype == np.uint8:
        out_img = img
    else:
        img = img.astype(np.float32)
        if img.max() > 1.1:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        img = np.power(img, 1.0 / 2.2)
        out_img = (img * 255.0).clip(0, 255).astype(np.uint8)

    if resize_factor is not None and resize_factor > 1:
        target_size = (max(1, out_img.shape[1] // resize_factor), max(1, out_img.shape[0] // resize_factor))
        out_img = cv.resize(out_img, target_size, interpolation=cv.INTER_AREA)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    return cv.imwrite(image_path, out_img)


def _export_illuminant_map_as_image(illuminant_map, image_path, mask=None):
    if illuminant_map is None:
        return False

    illuminant_map = np.asarray(illuminant_map, dtype=np.float32)
    if illuminant_map.ndim != 3 or illuminant_map.shape[2] != 2:
        return False

    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3 and mask_arr.shape[2] == 1:
            mask_arr = mask_arr[..., 0]
        mask_arr = mask_arr.astype(bool)
        if mask_arr.shape != illuminant_map.shape[:2]:
            try:
                mask_arr = cv.resize(mask_arr.astype(np.uint8), (illuminant_map.shape[1], illuminant_map.shape[0]), interpolation=cv.INTER_NEAREST).astype(bool)
            except Exception:
                mask_arr = np.ones(illuminant_map.shape[:2], dtype=bool)
    else:
        mask_arr = np.ones(illuminant_map.shape[:2], dtype=bool)

    rg = illuminant_map[..., 0]
    bg = illuminant_map[..., 1]
    height, width = rg.shape

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 0] = (rg - 1.0) * 0.5 + 0.5
    rgb[..., 1] = 1.0
    rgb[..., 2] = (bg - 1.0) * 0.5 + 0.5

    rgb = np.clip(rgb, 0.0, 1.0)

    if mask_arr.shape == (height, width):
        rgb[~mask_arr] = 0.0

    out_img = (rgb * 255.0).astype(np.uint8)
    # Convert RGB to BGR before saving with OpenCV
    out_img = cv.cvtColor(out_img, cv.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    return cv.imwrite(image_path, out_img)


def _prepare_export_paths(output_path, dataset_name, algo_name, variant_name, image_name):
    base_export_dir = os.path.splitext(output_path)[0] + "_exported_images"
    os.makedirs(base_export_dir, exist_ok=True)
    return base_export_dir


class Evaluator:
    def __init__(self, datasets, algorithms, camera=None, output_path="results.json", process_masked=False, num_workers=1, saturation_mask="none", color_checker="all", export_corrected_images=False, export_resize_factor=None):
        """
        Args:
            datasets: list of dataset name strings, e.g. ["gehler", "nus8"]
            algorithms: list of (algorithm_name, variant_name) tuples,
                        e.g. [("gray_world", "naive"), ("max_rgb", "99_percentile")]
            camera: if specified, only process images from this specific camera (optional)
            output_path: path for the output JSON file
            process_masked: if True, algorithms will exclude masked pixels (e.g. checkerboard)
            num_workers: number of processes to use (default: 1)
            saturation_mask: saturation mask string configuration for dataset providers
            color_checker: color checker configuration ("all" or "patch")
            export_corrected_images: if True, save corrected raw images to disk during evaluation
            export_resize_factor: optional integer downsample factor for exported corrected images (powers of two)
        """
        self.datasets = datasets
        self.algorithms = algorithms
        self.camera = camera
        self.output_path = output_path
        self.process_masked = process_masked
        self.num_workers = num_workers
        self.saturation_mask_str = saturation_mask
        self.color_checker_str = color_checker
        self.export_corrected_images = export_corrected_images
        self.export_resize_factor = export_resize_factor

        if self.export_resize_factor is not None:
            if self.export_resize_factor not in [2, 4, 8, 16, 32, 64]:
                raise ValueError("export_resize_factor must be a power of two, e.g. 2, 4, 8, 16, 32, 64")
        
        has_nus_datasets = any(ds in ("nus8", "nus8extended") for ds in datasets)
        has_gehler_dataset = any(ds == "gehler" for ds in datasets)

        if not process_masked:
            if saturation_mask != "none" or color_checker != "all":
                raise ValueError("saturation_mask and color_checker parameters are only valid when process_masked is enabled")

        if saturation_mask != "none" and not (has_nus_datasets or has_gehler_dataset):
            raise ValueError("saturation_mask is only valid for nus8, nus8extended, and gehler datasets")

        if color_checker != "all" and not (has_nus_datasets or has_gehler_dataset):
            raise ValueError("color_checker is only valid for nus8, nus8extended, and gehler datasets")
        
        saturation_masks = {
            'none': None,
            'raw_all_98': ('raw', 'all', 0.98),
            'raw_all_100': ('raw', 'all', 1.0),
            'raw_any_98': ('raw', 'any', 0.98),
            'raw_any_100': ('raw', 'any', 1.0),
            'normalized_all_98': ('normalized', 'all', 0.98),
            'normalized_all_100': ('normalized', 'all', 1.0),
            'normalized_any_98': ('normalized', 'any', 0.98),
            'normalized_any_100': ('normalized', 'any', 1.0),
        }
        if saturation_mask not in saturation_masks:
            raise ValueError(f"Invalid saturation mask: {saturation_mask}")
        self.saturation_mask_tuple = saturation_masks[saturation_mask]

        # Validate inputs
        for ds in datasets:
            if ds not in DATASET_PROVIDERS:
                raise ValueError(f"Unknown dataset: {ds}. Valid: {list(DATASET_PROVIDERS.keys())}")
        for algo, variant in algorithms:
            if (algo, variant) not in ALGORITHM_REGISTRY:
                raise ValueError(f"Unknown algorithm variant: ({algo}, {variant}). Valid: {list(ALGORITHM_REGISTRY.keys())}")

    def run(self):
        """Run all evaluations and save results to JSON."""
        all_results = []
        tasks = []

        print(f"\nConfiguration:")
        print(f"  Workers: {self.num_workers}")
        print(f"  Process Masked: {self.process_masked}")
        print(f"  Export Corrected Images: {self.export_corrected_images}")
        if self.export_corrected_images:
            print(f"  Export Resize Factor: {self.export_resize_factor}")
        if self.camera:
            print(f"  Camera Filter: {self.camera}")

        for dataset_name in self.datasets:
            if dataset_name in ("nus8", "nus8extended", "gehler"):
                data_provider = DATASET_PROVIDERS[dataset_name](saturation_mask=self.saturation_mask_tuple, color_checker=self.color_checker_str)
            else:
                data_provider = DATASET_PROVIDERS[dataset_name]()
            num_dataset_images = len(data_provider)
            
            # Identify indices matching the camera filter
            matching_indices = []
            for idx in range(num_dataset_images):
                camera = _extract_camera(dataset_name, data_provider, idx)
                if not self.camera or camera == self.camera:
                    matching_indices.append(idx)
            
            num_matching = len(matching_indices)
            print(f"Queuing Dataset: {dataset_name} ({num_matching}/{num_dataset_images} images match camera filter)")

            for algo_name, variant_name in self.algorithms:
                for idx in matching_indices:
                    tasks.append((dataset_name, algo_name, variant_name, idx, self.process_masked, self.camera, self.saturation_mask_str, self.color_checker_str, self.export_corrected_images, self.export_resize_factor, self.output_path))
        total_tasks = len(tasks)
        processed_count = 0

        print(f"Total tasks in queue: {total_tasks}")

        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(_worker_fn, task): task for task in tasks}
                for future in tqdm(as_completed(futures), total=total_tasks, desc="Evaluating", unit="task"):
                    result = future.result()
                    if result:
                        all_results.append(result)
                        if result.get("errors") is None and "error_message" in result:
                             print(f"\n    ERROR: {result['error_message']}")
        else:
            # Sequential execution
            for task in tqdm(tasks, desc="Evaluating", unit="task"):
                result = _worker_fn(task)
                if result:
                    all_results.append(result)
                    if result.get("errors") is None and "error_message" in result:
                         print(f"\n    ERROR: {result['error_message']}")

        output = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "datasets": self.datasets,
                "algorithms": [[a, v] for a, v in self.algorithms],
                "process_masked": self.process_masked,
                "saturation_mask": self.saturation_mask_str,
                "color_checker": self.color_checker_str,
                "export_corrected_images": self.export_corrected_images,
                "export_resize_factor": self.export_resize_factor,
            },
            "results": all_results,
        }

        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {self.output_path} ({len(all_results)} entries)")
        return output
