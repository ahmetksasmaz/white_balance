import argparse
import multiprocessing
from evaluator import Evaluator, DATASET_PROVIDERS, ALGORITHM_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description="Run white balance evaluation across datasets and algorithms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m run_evaluator --datasets gehler --algorithms gray_world:naive --output results.json
  python -m run_evaluator --datasets gehler nus8 --algorithms gray_world:naive max_rgb:99_percentile
  python -m run_evaluator --datasets gehler --algorithms all --output full_results.json
        """
    )
    parser.add_argument(
        '--datasets', nargs='+', required=True,
        help=f'Dataset names. Valid: {list(DATASET_PROVIDERS.keys())}'
    )
    parser.add_argument(
        '--algorithms', nargs='+', required=True,
        help='Algorithm:variant pairs (e.g. gray_world:naive). Use "all" to run all algorithms.'
    )
    parser.add_argument(
        '--output', type=str, default='results.json',
        help='Output JSON file path (default: results.json)'
    )
    parser.add_argument(
        '--camera', type=str, default=None,
        help='If specified, only process images from this specific camera (optional)'
    )
    parser.add_argument(
        '--workers', type=str, default='1',
        help='Number of worker processes to use (default: 1). Use "max" for (cpu_count - 2)'
    )
    parser.add_argument(
        '--process-masked', action='store_true', default=False,
        help='If set, algorithms will exclude masked pixels (e.g. checkerboard regions)'
    )
    args = parser.parse_args()

    # Parse algorithms
    if args.algorithms == ["all"]:
        algorithms = list(ALGORITHM_REGISTRY.keys())
    else:
        algorithms = []
        for entry in args.algorithms:
            if ':' not in entry:
                raise ValueError(f"Invalid algorithm format: '{entry}'. Use 'algorithm:variant' format.")
            algo, variant = entry.split(':', 1)
            algorithms.append((algo, variant))

    # Parse workers
    if args.workers == 'max':
        num_workers = max(1, multiprocessing.cpu_count() - 2)
    else:
        num_workers = int(args.workers)

    print(f"Datasets: {args.datasets}")
    print(f"Algorithms: {algorithms}")
    print(f"Output: {args.output}")
    print(f"Process masked: {args.process_masked}")

    evaluator = Evaluator(
        datasets=args.datasets,
        algorithms=algorithms,
        camera=args.camera,
        output_path=args.output,
        process_masked=args.process_masked,
        num_workers=num_workers,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
