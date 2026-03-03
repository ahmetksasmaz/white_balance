import argparse
from reporter import Reporter


def main():
    parser = argparse.ArgumentParser(
        description="Generate aggregate error reports from evaluator results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m run_reporter --input results.json --output report.json
        """
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to the evaluator results JSON file'
    )
    parser.add_argument(
        '--output', type=str, default='report.json',
        help='Output report JSON file path (default: report.json)'
    )
    args = parser.parse_args()

    reporter = Reporter(
        input_path=args.input,
        output_path=args.output,
    )
    reporter.generate()


if __name__ == "__main__":
    main()
