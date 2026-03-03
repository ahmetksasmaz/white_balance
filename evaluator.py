import json
import datetime
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

from datasets.cubepp.cubepp_dataprovider import CubePPDataProvider
from datasets.lsmi.lsmi_dataprovider import LSMIDataProvider
from datasets.gehler.gehler_dataprovider import GehlerDataProvider
from datasets.nus8.nus8_dataprovider import NUS8DataProvider
from datasets.nus8.nus8_extended_dataprovider import NUS8ExtendedDataProvider

from white_balance_algorithms.gray_world.gray_world_naive import GrayWorldNaive
from white_balance_algorithms.gray_world.gray_world_95_boundaries_all_channels import GrayWorld95BoundariesAllChannels
from white_balance_algorithms.gray_world.gray_world_95_boundaries_any_channel import GrayWorld95BoundariesAnyChannel

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
from white_balance_algorithms.shades_of_gray.shades_of_gray_masked_default import ShadesOfGrayMaskedDefault
from white_balance_algorithms.shades_of_gray.shades_of_gray_masked_p3 import ShadesOfGrayMaskedP3

from white_balance_algorithms.fast_awb.fast_awb_default import FastAWBDefault
from white_balance_algorithms.fast_awb.fast_awb_p6 import FastAWBP6


DATASET_PROVIDERS = {
    "cubepp": CubePPDataProvider,
    "lsmi": LSMIDataProvider,
    "gehler": GehlerDataProvider,
    "nus8": NUS8DataProvider,
    "nus8extended": NUS8ExtendedDataProvider,
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
    ("shades_of_gray", "masked_default"): ShadesOfGrayMaskedDefault,
    ("shades_of_gray", "masked_p3"): ShadesOfGrayMaskedP3,
    ("fast_awb", "default"): FastAWBDefault,
    ("fast_awb", "p6"): FastAWBP6,
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
    dataset_name, algo_name, variant_name, idx, process_masked, camera_filter = task_info
    try:
        # Instantiate inside worker to avoid pickle issues with some objects
        data_provider = DATASET_PROVIDERS[dataset_name]()
        algorithm = ALGORITHM_REGISTRY[(algo_name, variant_name)]()

        camera = _extract_camera(dataset_name, data_provider, idx)

        data = data_provider[idx]
        image_name = data.get_image_name()

        estimations = algorithm.estimate(data, process_masked=process_masked)
        error_metrics = data.compute_error_metrics(estimations)

        return {
            "dataset": dataset_name,
            "camera": camera,
            "image_name": image_name,
            "algorithm": algo_name,
            "variant": variant_name,
            "errors": _serialize_error_metrics(error_metrics),
        }
    except Exception as e:
        return {
            "dataset": dataset_name,
            "camera": "unknown",
            "image_name": f"index_{idx}",
            "algorithm": algo_name,
            "variant": variant_name,
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


class Evaluator:
    def __init__(self, datasets, algorithms, camera=None, output_path="results.json", process_masked=False, num_workers=1):
        """
        Args:
            datasets: list of dataset name strings, e.g. ["gehler", "nus8"]
            algorithms: list of (algorithm_name, variant_name) tuples,
                        e.g. [("gray_world", "naive"), ("max_rgb", "99_percentile")]
            camera: if specified, only process images from this specific camera (optional)
            output_path: path for the output JSON file
            process_masked: if True, algorithms will exclude masked pixels (e.g. checkerboard)
            num_workers: number of processes to use (default: 1)
        """
        self.datasets = datasets
        self.algorithms = algorithms
        self.camera = camera
        self.output_path = output_path
        self.process_masked = process_masked
        self.num_workers = num_workers

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
        if self.camera:
            print(f"  Camera Filter: {self.camera}")

        for dataset_name in self.datasets:
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
                    tasks.append((dataset_name, algo_name, variant_name, idx, self.process_masked, self.camera))

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
            },
            "results": all_results,
        }

        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {self.output_path} ({len(all_results)} entries)")
        return output
