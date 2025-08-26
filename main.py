import numpy as np
import cv2 as cv
import sys
import os
import argparse

def main():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White Balance Algorithm Tester")
    parser.add_argument("--datasets", type=str, required=False, default="", help="List of the name of the datasets to run [cubepp], type 'all' for all")
    parser.add_argument("--algorithms", type=str, required=False, default="", help="List of the algorithms to run [gray_world], type 'all' for all")
    parser.add_argument("--metrics", type=str, required=False, default="", help="List of the metrics to use [mae, mse, e2000], type 'all' for all")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--export", type=str, required=False, default="", help="List of export images [algorithm, gt, merged, merged_resized], type 'all' for all")
    parser.add_argument("--skip", action="store_true", help="Skip already processed images with that algorithm (scans the output directory for formatted filenames)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the results after processing")
    args = parser.parse_args()

    dataset_names = args.datasets.split(",")
    algorithm_names = args.algorithms.split(",")
    metric_names = args.metrics.split(",")
    output_directory = args.output
    export_names = args.export.split(",")
    skip_processed = args.skip
    evaluate_results = args.evaluate