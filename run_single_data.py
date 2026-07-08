import argparse
import numpy as np

from evaluator import DATASET_PROVIDERS, ALGORITHM_REGISTRY


def run_single_data(dataset_name, index, algorithm_name, variant_name, process_masked=False, saturation_mask_str='none', color_checker_str='all'):
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

    algorithm_key = (algorithm_name, variant_name)
    if algorithm_key not in ALGORITHM_REGISTRY:
        raise ValueError(f"Invalid algorithm variant: {algorithm_key}")
    algorithm = ALGORITHM_REGISTRY[algorithm_key]()

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
    run_single_data(args.dataset, args.index, args.algorithm, args.variant, args.process_masked, args.saturation_mask, args.color_checker)


if __name__ == "__main__":
    main()
