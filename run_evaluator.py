import argparse
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

    print(f"Datasets: {args.datasets}")
    print(f"Algorithms: {algorithms}")
    print(f"Output: {args.output}")

    evaluator = Evaluator(
        datasets=args.datasets,
        algorithms=algorithms,
        output_path=args.output,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
