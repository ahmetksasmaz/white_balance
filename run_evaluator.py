import argparse
import json
import multiprocessing
import os
import re
from evaluator import Evaluator, DATASET_PROVIDERS, ALGORITHM_REGISTRY
from reporter import Reporter


def load_configuration(config_path):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        raw_text = f.read()

    # Allow JSON configuration files to include comments with //, #, or /* */
    without_line_comments = re.sub(r'(?m)^\s*(//|#).*$', '', raw_text)
    without_block_comments = re.sub(r'/\*.*?\*/', '', without_line_comments, flags=re.DOTALL)
    return json.loads(without_block_comments)


def get_config_value(args, config, key, default=None):
    value = getattr(args, key)
    if value is not None:
        return value
    return config.get(key, default)


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y')
    return bool(value)


def main():
    parser = argparse.ArgumentParser(
        description="Run white balance evaluation across datasets and algorithms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m run_evaluator
  python -m run_evaluator --config configuration.json
  python -m run_evaluator --datasets gehler --algorithms gray_world:naive --output results.json
  python -m run_evaluator --datasets gehler nus8 --algorithms gray_world:naive max_rgb:99_percentile
  python -m run_evaluator --datasets gehler --algorithms all --output full_results.json

Command-line arguments override values from configuration.json.
        """
    )
    parser.add_argument(
        '--config', type=str, default='configuration.json',
        help='Path to the configuration JSON file (default: configuration.json)'
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help=f'Dataset names. Valid: {list(DATASET_PROVIDERS.keys())}. If omitted, loaded from configuration file.'
    )
    parser.add_argument(
        '--algorithms', nargs='+', default=None,
        help='Algorithm:variant pairs (e.g. gray_world:naive). Use "all" to run all algorithms. If omitted, loaded from configuration file.'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON file path (default: results.json or configuration file value)'
    )
    parser.add_argument(
        '--camera', type=str, default=None,
        help='If specified, only process images from this specific camera (optional)'
    )
    parser.add_argument(
        '--workers', type=str, default=None,
        help='Number of worker processes to use (default: 1). Use "max" for (cpu_count - 2). If omitted, loaded from configuration file.'
    )
    parser.add_argument(
        '--process-masked', action='store_true', default=None,
        help='If set, algorithms will exclude masked pixels (e.g. checkerboard regions). If omitted, loaded from configuration file.'
    )
    parser.add_argument(
        '--saturation-mask', type=str, default=None, choices=[
            'none', 'raw_all_98', 'raw_all_100', 'raw_any_98', 'raw_any_100', 
            'normalized_all_98', 'normalized_all_100', 'normalized_any_98', 'normalized_any_100'
        ], help='Saturation mask configuration. If omitted, loaded from configuration file'
    )
    parser.add_argument(
        '--color-checker', type=str, default=None, choices=['all', 'patch'], 
        help='Color checker mask type. If omitted, loaded from configuration file'
    )
    parser.add_argument(
        '--export-corrected-images', action='store_true', default=None,
        help='Export corrected raw images and estimated sRGB images to disk during evaluation. If omitted, loaded from configuration file.'
    )
    parser.add_argument(
        '--export-resize-factor', type=int, default=None, choices=[2, 4, 8, 16, 32, 64],
        help='Optional downsample factor for exported corrected images (powers of two). If omitted, loaded from configuration file.'
    )
    args = parser.parse_args()
    config = load_configuration(args.config)

    datasets = get_config_value(args, config, 'datasets')
    algorithms = get_config_value(args, config, 'algorithms')
    output = get_config_value(args, config, 'output', 'results.json')
    camera = get_config_value(args, config, 'camera')
    workers = get_config_value(args, config, 'workers', '1')
    process_masked = normalize_bool(get_config_value(args, config, 'process_masked', False))
    saturation_mask = get_config_value(args, config, 'saturation_mask', 'none')
    color_checker = get_config_value(args, config, 'color_checker', 'all')
    export_corrected_images = normalize_bool(get_config_value(args, config, 'export_corrected_images', False))
    export_resize_factor = get_config_value(args, config, 'export_resize_factor', None)

    if datasets is None:
        raise ValueError('Dataset list must be provided either in configuration.json or via --datasets')
    if algorithms is None:
        raise ValueError('Algorithm list must be provided either in configuration.json or via --algorithms')

    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    if algorithms == ['all']:
        algorithms = list(ALGORITHM_REGISTRY.keys())
    else:
        algo_pairs = []
        for entry in algorithms:
            if ':' not in entry:
                raise ValueError(f"Invalid algorithm format: '{entry}'. Use 'algorithm:variant' format.")
            algo, variant = entry.split(':', 1)
            algo_pairs.append((algo, variant))
        algorithms = algo_pairs

    if isinstance(workers, int):
        num_workers = workers
    elif isinstance(workers, str) and workers.lower() == 'max':
        num_workers = max(1, multiprocessing.cpu_count() - 2)
    else:
        num_workers = int(workers)

    print(f"Configuration file: {args.config}")
    print(f"Datasets: {datasets}")
    print(f"Algorithms: {algorithms}")
    print(f"Output: {output}")
    print(f"Process masked: {process_masked}")
    print(f"Saturation mask: {saturation_mask}")
    print(f"Color checker: {color_checker}")
    print(f"Export corrected images: {export_corrected_images}")
    print(f"Export resize factor: {export_resize_factor}")

    evaluator = Evaluator(
        datasets=datasets,
        algorithms=algorithms,
        camera=camera,
        output_path=output,
        process_masked=process_masked,
        num_workers=num_workers,
        saturation_mask=saturation_mask,
        color_checker=color_checker,
        export_corrected_images=export_corrected_images,
        export_resize_factor=export_resize_factor,
    )
    evaluator.run()

    report_filename = os.path.splitext(output)[0] + "_report.json"
    reporter = Reporter(input_path=output, output_path=report_filename)
    reporter.generate()


if __name__ == "__main__":
    main()
