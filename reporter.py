import json
import datetime
import numpy as np
from collections import defaultdict


class Reporter:
    def __init__(self, input_path, output_path="report.json"):
        self.input_path = input_path
        self.output_path = output_path

    def _compute_statistics(self, values):
        arr = np.array(values, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        n = len(valid)
        if n == 0:
            return {"mean": None, "median": None, "trimean": None,
                    "best_25": None, "worst_25": None, "max": None}

        arr_sorted = np.sort(valid)
        q1     = float(np.percentile(valid, 25))
        median = float(np.median(valid))
        q3     = float(np.percentile(valid, 75))
        trimean = (q1 + 2.0 * median + q3) / 4.0

        best_25_count  = max(1, n // 4)
        worst_25_count = max(1, n // 4)

        nan_count = int(np.sum(~np.isfinite(arr)))
        result = {
            "mean":     float(np.mean(valid)),
            "median":   median,
            "trimean":  trimean,
            "best_25":  float(np.mean(arr_sorted[:best_25_count])),
            "worst_25": float(np.mean(arr_sorted[-worst_25_count:])),
            "max":      float(np.max(valid)),
        }
        if nan_count > 0:
            result["nan_count"] = nan_count
        return result

    def generate(self):
        with open(self.input_path, "r") as f:
            data = json.load(f)

        results = data.get("results", [])
        groups = defaultdict(lambda: defaultdict(list))

        for entry in results:
            if entry.get("errors") is None:
                continue

            key = (entry["dataset"], entry["camera"], entry["algorithm"], entry["variant"])
            dataset_key = (entry["dataset"], "all", entry["algorithm"], entry["variant"])
            errors = entry["errors"]

            for error_key in ("single_illuminant_errors", "image_errors", "multi_illuminant_errors"):
                if errors.get(error_key) is not None:
                    for metric_name, metric_value in errors[error_key].items():
                        if metric_value is not None:
                            groups[key][metric_name].append(metric_value)
                            groups[dataset_key][metric_name].append(metric_value)

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

    @staticmethod
    def _fmt(v):
        return f"{v:>8.3f}" if v is not None else f"{'N/A':>8}"

    def _print_summary(self, reports):
        print(f"\n{'='*100}")
        print(f"{'Dataset':<12} {'Camera':<20} {'Algorithm':<16} {'Variant':<28} {'#Imgs':>6}  {'Metric':<16} {'Mean':>8} {'Median':>8} {'Tri-M':>8} {'B25%':>8} {'W25%':>8} {'Max':>8}")
        print(f"{'-'*100}")

        f = self._fmt
        for r in reports:
            first_metric = True
            for metric_name, stats in r["metrics"].items():
                vals = f"{f(stats['mean'])} {f(stats['median'])} {f(stats['trimean'])} {f(stats['best_25'])} {f(stats['worst_25'])} {f(stats['max'])}"
                if first_metric:
                    print(f"{r['dataset']:<12} {r['camera']:<20} {r['algorithm']:<16} {r['variant']:<28} {r['num_images']:>6}  {metric_name:<16} {vals}")
                    first_metric = False
                else:
                    print(f"{'':.<12} {'':.<20} {'':.<16} {'':.<28} {'':>6}  {metric_name:<16} {vals}")

        print(f"{'='*100}")
