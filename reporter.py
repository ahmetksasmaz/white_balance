import json
import datetime
import numpy as np
from collections import defaultdict


class Reporter:
    def __init__(self, input_path, output_path="report.json"):
        """
        Args:
            input_path: path to the evaluator JSON results file
            output_path: path for the output report JSON file
        """
        self.input_path = input_path
        self.output_path = output_path

    def _compute_statistics(self, values):
        """Compute aggregate statistics for a list of numeric values."""
        arr = np.array(values, dtype=np.float64)
        arr_sorted = np.sort(arr)
        n = len(arr_sorted)

        q1 = float(np.percentile(arr, 25))
        median = float(np.median(arr))
        q3 = float(np.percentile(arr, 75))
        trimean = (q1 + 2.0 * median + q3) / 4.0

        # Best 25%: mean of the lowest quartile
        best_25_count = max(1, n // 4)
        best_25 = float(np.mean(arr_sorted[:best_25_count]))

        # Worst 25%: mean of the highest quartile
        worst_25_count = max(1, n // 4)
        worst_25 = float(np.mean(arr_sorted[-worst_25_count:]))

        return {
            "mean": float(np.mean(arr)),
            "median": float(median),
            "trimean": float(trimean),
            "best_25": best_25,
            "worst_25": worst_25,
            "max": float(np.max(arr)),
        }

    def generate(self):
        """Parse results and generate aggregated report."""
        with open(self.input_path, "r") as f:
            data = json.load(f)

        results = data.get("results", [])

        # Group by (dataset, camera, algorithm, variant) -> {metric_name: [values]}
        groups = defaultdict(lambda: defaultdict(list))

        for entry in results:
            if entry.get("errors") is None:
                continue  # skip failed entries

            key = (
                entry["dataset"],
                entry["camera"],
                entry["algorithm"],
                entry["variant"],
            )

            # Also aggregate across all cameras per dataset
            dataset_key = (
                entry["dataset"],
                "all",
                entry["algorithm"],
                entry["variant"],
            )

            errors = entry["errors"]

            # Collect single illuminant errors
            if errors.get("single_illuminant_errors") is not None:
                for metric_name, metric_value in errors["single_illuminant_errors"].items():
                    if metric_value is not None:
                        groups[key][metric_name].append(metric_value)
                        groups[dataset_key][metric_name].append(metric_value)

            # Collect image errors
            if errors.get("image_errors") is not None:
                for metric_name, metric_value in errors["image_errors"].items():
                    if metric_value is not None:
                        groups[key][metric_name].append(metric_value)
                        groups[dataset_key][metric_name].append(metric_value)

            # Collect multi illuminant errors (when implemented)
            if errors.get("multi_illuminant_errors") is not None:
                for metric_name, metric_value in errors["multi_illuminant_errors"].items():
                    if metric_value is not None:
                        groups[key][metric_name].append(metric_value)
                        groups[dataset_key][metric_name].append(metric_value)

        # Build reports
        reports = []
        for (dataset, camera, algorithm, variant), metrics_dict in sorted(groups.items(), key=lambda x: x[0]):
            report_entry = {
                "dataset": dataset,
                "camera": camera,
                "algorithm": algorithm,
                "variant": variant,
                "num_images": 0,
                "metrics": {},
            }

            for metric_name, values in metrics_dict.items():
                report_entry["num_images"] = max(report_entry["num_images"], len(values))
                report_entry["metrics"][metric_name] = self._compute_statistics(values)

            reports.append(report_entry)

        output = {
            "metadata": {
                "source_file": self.input_path,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "reports": reports,
        }

        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Report saved to {self.output_path} ({len(reports)} groups)")
        self._print_summary(reports)
        return output

    def _print_summary(self, reports):
        """Print a human-readable summary table to stdout."""
        print(f"\n{'='*100}")
        print(f"{'Dataset':<12} {'Camera':<20} {'Algorithm':<16} {'Variant':<28} {'#Imgs':>6}  {'Metric':<16} {'Mean':>8} {'Median':>8} {'Tri-M':>8} {'B25%':>8} {'W25%':>8} {'Max':>8}")
        print(f"{'-'*100}")

        for r in reports:
            first_metric = True
            for metric_name, stats in r["metrics"].items():
                if first_metric:
                    print(f"{r['dataset']:<12} {r['camera']:<20} {r['algorithm']:<16} {r['variant']:<28} {r['num_images']:>6}  {metric_name:<16} {stats['mean']:>8.3f} {stats['median']:>8.3f} {stats['trimean']:>8.3f} {stats['best_25']:>8.3f} {stats['worst_25']:>8.3f} {stats['max']:>8.3f}")
                    first_metric = False
                else:
                    print(f"{'':.<12} {'':.<20} {'':.<16} {'':.<28} {'':>6}  {metric_name:<16} {stats['mean']:>8.3f} {stats['median']:>8.3f} {stats['trimean']:>8.3f} {stats['best_25']:>8.3f} {stats['worst_25']:>8.3f} {stats['max']:>8.3f}")

        print(f"{'='*100}")
