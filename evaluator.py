import json
import datetime
import traceback

from datasets.cubepp.cubepp_dataprovider import CubePPDataProvider
from datasets.lsmi.lsmi_dataprovider import LSMIDataProvider
from datasets.gehler.gehler_dataprovider import GehlerDataProvider
from datasets.nus8.nus8_dataprovider import NUS8DataProvider

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

from white_balance_algorithms.fast_awb.fast_awb_default import FastAWBDefault
from white_balance_algorithms.fast_awb.fast_awb_p6 import FastAWBP6


DATASET_PROVIDERS = {
    "cubepp": CubePPDataProvider,
    "lsmi": LSMIDataProvider,
    "gehler": GehlerDataProvider,
    "nus8": NUS8DataProvider,
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

    if dataset_name == "nus8":
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
    def __init__(self, datasets, algorithms, output_path="results.json", process_masked=False):
        """
        Args:
            datasets: list of dataset name strings, e.g. ["gehler", "nus8"]
            algorithms: list of (algorithm_name, variant_name) tuples,
                        e.g. [("gray_world", "naive"), ("max_rgb", "99_percentile")]
            output_path: path for the output JSON file
            process_masked: if True, algorithms will exclude masked pixels (e.g. checkerboard)
        """
        self.datasets = datasets
        self.algorithms = algorithms
        self.output_path = output_path
        self.process_masked = process_masked

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

        for dataset_name in self.datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")

            data_provider = DATASET_PROVIDERS[dataset_name]()
            num_images = len(data_provider)
            print(f"  Total images: {num_images}")

            for algo_name, variant_name in self.algorithms:
                print(f"\n  Algorithm: {algo_name}/{variant_name}")
                algorithm = ALGORITHM_REGISTRY[(algo_name, variant_name)]()

                for idx in range(num_images):
                    try:
                        data = data_provider[idx]
                        image_name = data.get_image_name()
                        camera = _extract_camera(dataset_name, data_provider, idx)

                        estimations = algorithm.estimate(data, process_masked=self.process_masked)
                        error_metrics = data.compute_error_metrics(estimations)

                        result_entry = {
                            "dataset": dataset_name,
                            "camera": camera,
                            "image_name": image_name,
                            "algorithm": algo_name,
                            "variant": variant_name,
                            "errors": _serialize_error_metrics(error_metrics),
                        }
                        all_results.append(result_entry)

                        if (idx + 1) % 50 == 0 or idx == num_images - 1:
                            print(f"    Processed {idx + 1}/{num_images} images")

                    except Exception as e:
                        print(f"    ERROR on image index {idx}: {e}")
                        traceback.print_exc()
                        all_results.append({
                            "dataset": dataset_name,
                            "camera": _extract_camera(dataset_name, data_provider, idx),
                            "image_name": f"index_{idx}",
                            "algorithm": algo_name,
                            "variant": variant_name,
                            "errors": None,
                            "error_message": str(e),
                        })

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
