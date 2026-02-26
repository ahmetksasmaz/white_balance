import argparse
import cv2 as cv
import numpy as np
from datasets.data import Data
from datasets.cubepp.cubepp_dataprovider import CubePPDataProvider
from datasets.lsmi.lsmi_dataprovider import LSMIDataProvider

from white_balance_algorithms.gray_world.gray_world_naive import GrayWorldNaive
from white_balance_algorithms.gray_world.gray_world_boundaries_all_channels import GrayWorldBoundariesAllChannels
from white_balance_algorithms.gray_world.gray_world_boundaries_any_channel import GrayWorldBoundariesAnyChannel

from white_balance_algorithms.max_rgb.max_rgb_naive import MaxRGBNaive
from white_balance_algorithms.max_rgb.max_rgb_percentile import MaxRGBPercentile
from white_balance_algorithms.max_rgb.max_rgb_gaussian import MaxRGBGaussian

from white_balance_algorithms.shades_of_gray.shades_of_gray_default import ShadesOfGrayDefault

from white_balance_algorithms.fast_awb.fast_awb_default import FastAWBDefault

def run_single_data(dataset_name, index, algorithm_name, variant_name, params):
    if dataset_name == "cubepp":
        data_provider = CubePPDataProvider()
    elif dataset_name == "lsmi":
        data_provider = LSMIDataProvider()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    data = data_provider[index]

    if algorithm_name == "gray_world":
        if variant_name == "naive":
            algorithm = GrayWorldNaive()
        elif variant_name == "boundaries_any_channel":
            algorithm = GrayWorldBoundariesAnyChannel()
        elif variant_name == "boundaries_all_channels":
            algorithm = GrayWorldBoundariesAllChannels()
        else:
            raise ValueError(f"Invalid variant name for gray_world: {variant_name}")
    elif algorithm_name == "max_rgb":
        if variant_name == "naive":
            algorithm = MaxRGBNaive()
        elif variant_name == "percentile":
            percentile = float(params[0]) if params else 99
            algorithm = MaxRGBPercentile(percentile=percentile)
        elif variant_name == "gaussian":
            kernel_size = int(params[0]) if params else 5
            sigma = float(params[1]) if len(params) > 1 else 1
            algorithm = MaxRGBGaussian(kernel_size=kernel_size, sigma=sigma)
        else:
            raise ValueError(f"Invalid variant name for max_rgb: {variant_name}")
    elif algorithm_name == "shades_of_gray":
        if variant_name == "default":
            p = float(params[0]) if params else 6
            algorithm = ShadesOfGrayDefault(p=p)
        else:
            raise ValueError(f"Invalid variant name for shades_of_gray: {variant_name}")
    elif algorithm_name == "fast_awb":
        if variant_name == "default":
            algorithm = FastAWBDefault()
        else:
            raise ValueError(f"Invalid variant name for fast_awb: {variant_name}")
    else:
        raise ValueError(f"Invalid algorithm name: {algorithm_name}")

    estimated_illuminant = algorithm.estimate(data)
    print(f"Estimated illuminant (r/g, b/g): {estimated_illuminant}")

    error_metrics = data.compute_error_metrics([estimated_illuminant])
    print(f"Error metrics: {error_metrics}")

def main():
    parser = argparse.ArgumentParser(description="Run single data white balance experiment.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--index', type=int, required=True, help='Image index')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm name')
    parser.add_argument('--variant', type=str, required=True, help='Algorithm variant')
    parser.add_argument('--params', nargs='*', default=[], help='Algorithm parameter list (key=value pairs)')

    args = parser.parse_args()

    valid_datasets = ["cubepp", "lsmi"]
    valid_algorithms = ["gray_world", "max_rgb", "shades_of_gray", "fast_awb"]
    valid_variants = {
        "gray_world": ["naive", "boundaries_all_channels", "boundaries_any_channel"],
        "max_rgb": ["naive", "percentile", "gaussian"],
        "shades_of_gray": ["default"],
        "fast_awb": ["default"]
    }

    if args.dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset: {args.dataset}. Valid options: {valid_datasets}")
    if args.algorithm not in valid_algorithms:
        raise ValueError(f"Invalid algorithm: {args.algorithm}. Valid options: {valid_algorithms}")
    if args.variant not in valid_variants[args.algorithm]:
        raise ValueError(f"Invalid variant: {args.variant} for algorithm {args.algorithm}. Valid options: {valid_variants[args.algorithm]}")

    print(f"Dataset: {args.dataset}", f"Index: {args.index}", f"Algorithm: {args.algorithm}", f"Variant: {args.variant}", f"Params: {args.params}")

    run_single_data(args.dataset, args.index, args.algorithm, args.variant, args.params)

if __name__ == "__main__":
    main()
