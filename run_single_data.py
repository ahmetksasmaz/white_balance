import argparse
import numpy as np
from datasets.cubepp.cubepp_dataprovider import CubePPDataProvider
from datasets.lsmi.lsmi_dataprovider import LSMIDataProvider
from datasets.gehler.gehler_dataprovider import GehlerDataProvider
from datasets.nus8.nus8_dataprovider import NUS8DataProvider
from datasets.nus8.nus8_extended_dataprovider import NUS8ExtendedDataProvider

from evaluator import DATASET_PROVIDERS, ALGORITHM_REGISTRY

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

from white_balance_algorithms.fast_awb.fast_awb_default import FastAWBDefault
from white_balance_algorithms.fast_awb.fast_awb_p6 import FastAWBP6

from white_balance_algorithms.cheng.cheng_prc_0_5 import ChengPrc05
from white_balance_algorithms.cheng.cheng_prc_3 import ChengPrc3

def run_single_data(dataset_name, index, algorithm_name, variant_name, process_masked=False, saturation_mask_str='none', color_checker_str='all', input_resize_factor=None):
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

    if dataset_name not in DATASET_PROVIDERS:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    if dataset_name in ("nus8", "nus8extended", "gehler"):
        data_provider = DATASET_PROVIDERS[dataset_name](saturation_mask=saturation_mask_tuple, color_checker=color_checker_str)
    else:
        data_provider = DATASET_PROVIDERS[dataset_name]()

    data = data_provider[index]
    data.resize(input_resize_factor)

    if algorithm_name == "gray_world":
        if variant_name == "naive":
            algorithm = GrayWorldNaive()
        elif variant_name == "95_boundaries_any_channel":
            algorithm = GrayWorld95BoundariesAnyChannel()
        elif variant_name == "95_boundaries_all_channels":
            algorithm = GrayWorld95BoundariesAllChannels()
        else:
            raise ValueError(f"Invalid variant name for gray_world: {variant_name}")
    elif algorithm_name == "max_rgb":
        if variant_name == "naive":
            algorithm = MaxRGBNaive()
        elif variant_name == "99_percentile":
            algorithm = MaxRGB99Percentile()
        elif variant_name == "95_percentile":
            algorithm = MaxRGB95Percentile()
        elif variant_name == "gaussian":
            algorithm = MaxRGBGaussian()
        elif variant_name == "gaussian_95_percentile":
            algorithm = MaxRGBGaussian95Percentile()
        elif variant_name == "gaussian_99_percentile":
            algorithm = MaxRGBGaussian99Percentile()
        elif variant_name == "median":
            algorithm = MaxRGBMedian()
        elif variant_name == "median_95_percentile":
            algorithm = MaxRGBMedian95Percentile()
        elif variant_name == "median_99_percentile":
            algorithm = MaxRGBMedian99Percentile()
        else:
            raise ValueError(f"Invalid variant name for max_rgb: {variant_name}")
    elif algorithm_name == "shades_of_gray":
        if variant_name == "default":
            algorithm = ShadesOfGrayDefault()
        elif variant_name == "p3":
            algorithm = ShadesOfGrayP3()
        elif variant_name == "p4":
            algorithm = ShadesOfGrayP4()
        else:
            raise ValueError(f"Invalid variant name for shades_of_gray: {variant_name}")
    elif algorithm_name == "fast_awb":
        if variant_name == "default":
            algorithm = FastAWBDefault()
        elif variant_name == "p6":
            algorithm = FastAWBP6()
        else:
            raise ValueError(f"Invalid variant name for fast_awb: {variant_name}")
    elif algorithm_name == "cheng":
        if variant_name == "prc_0_5":
            algorithm = ChengPrc05()
        elif variant_name == "prc_3":
            algorithm = ChengPrc3()
        else:
            raise ValueError(f"Invalid variant name for cheng: {variant_name}")
    else:
        raise ValueError(f"Invalid algorithm name: {algorithm_name}")

    estimations = algorithm.estimate(data, process_masked=process_masked)
    print(f"Estimated illuminant (r/g, b/g): {estimations['single_illuminant']}")
    print(f"Estimated multi-illuminants: {estimations['multi_illuminants']}")
    print(f"Estimated illuminant map: {estimations['illuminant_map']}")
    print(f"Estimated sRGB image: {estimations['estimated_srgb_image']}")

    error_metrics = data.compute_error_metrics(estimations)
    print(f"Error metrics: {error_metrics}")


def main():
    parser = argparse.ArgumentParser(description="Run single data white balance experiment.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--index', type=int, required=True, help='Image index')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm name')
    parser.add_argument('--variant', type=str, required=True, help='Algorithm variant')
    parser.add_argument('--process-masked', action='store_true', default=False, help='Exclude masked pixels')
    parser.add_argument('--input-resize-factor', type=int, default=None, choices=[2, 4, 8, 16, 32, 64], help='Downsample factor applied to the loaded input before evaluation')
    parser.add_argument('--saturation-mask', type=str, default='none', choices=[
        'none', 'raw_all_98', 'raw_all_100', 'raw_any_98', 'raw_any_100',
        'normalized_all_98', 'normalized_all_100', 'normalized_any_98', 'normalized_any_100'
    ], help='Saturation mask configuration')
    parser.add_argument('--color-checker', type=str, default='all', choices=['all', 'patch'], help='Color checker mask type')
    args = parser.parse_args()

    if not args.process_masked:
        if args.saturation_mask != 'none' or args.color_checker != 'all':
            raise ValueError("saturation-mask and color-checker parameters are only valid when process-masked is enabled")

    if args.saturation_mask != 'none' and args.dataset not in ['nus8', 'nus8extended', 'gehler']:
        raise ValueError("saturation-mask parameter is only valid for nus8, nus8extended, and gehler datasets")

    if args.color_checker != 'all' and args.dataset not in ['nus8', 'nus8extended', 'gehler']:
        raise ValueError("color-checker parameter is only valid for nus8, nus8extended, or gehler datasets")

    print(f"Dataset: {args.dataset}", f"Index: {args.index}", f"Algorithm: {args.algorithm}", f"Variant: {args.variant}")

    run_single_data(args.dataset, args.index, args.algorithm, args.variant, args.process_masked, args.saturation_mask, args.color_checker, args.input_resize_factor)


if __name__ == "__main__":
    main()
