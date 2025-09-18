import numpy as np
import cv2 as cv
import sys
import os
import argparse
from evaluator import *
from configuration import *
from datasets.cubepp.cubepp_dataloader import CubePPDataLoader
from white_balance_algorithms.gray_world import GrayWorld
from white_balance_algorithms.max_rgb import MaxRGB
from white_balance_algorithms.deep_wb import DeepWB
from helper import *
from errors import create_error_heatmap
import tqdm

def main(dataset_names, algorithm_names, output_directory, export_types, skip_processed, evaluate_only, export_resized):
    evaluator = Evaluator(os.path.join(output_directory, "results.csv"), os.path.join(output_directory, "evaluation.csv"), export_period=100)

    if not evaluate_only:
        # Load database
        # Run every image for every algorithm
        # For every image-algorithm pair, compute every metric, export every type
        for dataset_name in dataset_names:
            dataloader = None
            if dataset_name == "cubepp":
                dataloader = CubePPDataLoader()
            for algorithm_name in algorithm_names:
                algorithm = None
                if algorithm_name == "gray_world":
                    algorithm = GrayWorld()
                elif algorithm_name == "max_rgb":
                    algorithm = MaxRGB()
                elif algorithm_name == "deep_wb":
                    algorithm = DeepWB(model_path="/white_balance_algorithms/internal/deepwb/models/net_awb.pth", max_dim=656)
                print(f"Processing dataset {dataset_name} with algorithm {algorithm_name}")
                for i in tqdm.tqdm(range(len(dataloader))):
                    data = dataloader[i]
                    orig_image = data.get_image()
                    gt_image = data.get_gt_image()
                    if gt_image is None:
                        continue
                    if skip_processed and evaluator.has_processed(dataset_name, data.get_image_name(), algorithm_name):
                        continue
                    adapted_image, metadata = algorithm.apply(orig_image)
                    heatmap_image, mean_error = create_error_heatmap(adapted_image, gt_image)
                    orig_h, orig_w = orig_image.shape[:2]
                    new_w = int(orig_w * (512 / orig_h))
                    evaluator.update_processed(dataset_name, data.get_image_name(), algorithm_name, {"de2000": mean_error})
                    postfix = "full"
                    if export_resized:
                        orig_image = cv.resize(orig_image, (new_w, 512))
                        adapted_image = cv.resize(adapted_image, (new_w, 512))
                        gt_image = cv.resize(gt_image, (new_w, 512))
                        heatmap_image = cv.resize(heatmap_image, (new_w, 512))
                        postfix = "resized"
                    if "algorithm" in export_types:
                        export_path = os.path.join(output_directory, "exports", f"{data.get_image_name()}_{dataset_name}_{algorithm_name}_algorithm_{postfix}.png")
                        adapted_image_display = prepare_display(adapted_image, correct_gamma=True)
                        cv.imwrite(export_path, adapted_image_display)
                    if "gt" in export_types:
                        export_path = os.path.join(output_directory, "exports", f"{data.get_image_name()}_{dataset_name}_gt_{postfix}.png")
                        if not os.path.exists(export_path):
                            gt_image_display = prepare_display(gt_image, correct_gamma=True)
                            cv.imwrite(export_path, gt_image_display)
                    if "heatmap" in export_types:
                        export_path = os.path.join(output_directory, "exports", f"{data.get_image_name()}_{dataset_name}_{algorithm_name}_heatmap_{postfix}.png")
                        heatmap_image_display = prepare_display(heatmap_image, correct_gamma=False)
                        cv.imwrite(export_path, heatmap_image_display)
    evaluator.evaluate()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White Balance Algorithm Tester")
    parser.add_argument("--datasets", type=str, required=False, default="all", help="List of the name of the datasets to run [cubepp], type 'all' for all")
    parser.add_argument("--algorithms", type=str, required=False, default="all", help="List of the algorithms to run [gray_world, max_rgb, deep_wb], type 'all' for all")
    parser.add_argument("--output", type=str, required=True, default="output", help="Path to the output directory")
    parser.add_argument("--export", type=str, required=False, default="none", help="List of export images [none, algorithm, gt, heatmap], type 'all' for all")
    parser.add_argument("--export_resized", action="store_true", help="Export resized images (height 512px)")
    parser.add_argument("--skip", action="store_true", help="Skip already processed images with that algorithm (scans the output directory for formatted filenames)")
    parser.add_argument("--evaluate_only", action="store_true", help="Only evaluate the results already processed")
    args = parser.parse_args()

    dataset_names = args.datasets.split(",")
    algorithm_names = args.algorithms.split(",")
    output_directory = args.output
    export_types = args.export.split(",")
    export_resized = args.export_resized
    skip_processed = args.skip
    evaluate_only = args.evaluate_only

    if "all" in dataset_names:
        dataset_names = VALID_DATASETS
    if "all" in algorithm_names:
        algorithm_names = VALID_ALGORITHMS
    if "none" in export_types:
        export_types = []
    if "all" in export_types:
        export_types = VALID_EXPORTS
    
    for dataset in dataset_names:
        if dataset not in VALID_DATASETS:
            print(f"Invalid dataset: {dataset}")
            sys.exit(1)
    for algo in algorithm_names:
        if algo not in VALID_ALGORITHMS:
            print(f"Invalid algorithm: {algo}")
            sys.exit(1)
    for export in export_types:
        if export not in VALID_EXPORTS:
            print(f"Invalid export: {export}")
            sys.exit(1)
    if dataset_names == [] or algorithm_names == []:
        print("No datasets or algorithms to process")
        sys.exit(1)

    # Try to create output directory, if it fails exit
    try:
        os.makedirs(output_directory, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory: {e}")
        sys.exit(1)

    # Try to create results.csv in output directory if not exists
    results_file = os.path.join(output_directory, "results.csv")
    if not os.path.exists(results_file):
        try:
            file = open(results_file, "w")
            file.write("")
            file.close()
        except Exception as e:
            print(f"Failed to create results file: {e}")
            sys.exit(1)
    evaluation_file = os.path.join(output_directory, "evaluation.csv")
    if not os.path.exists(evaluation_file):
        try:
            file = open(evaluation_file, "w")
            file.write("")
            file.close()
        except Exception as e:
            print(f"Failed to create evaluation file: {e}")
            sys.exit(1)

    # Try to create exports directory in output directory if not exists
    exports_dir = os.path.join(output_directory, "exports")
    if not os.path.exists(exports_dir):
        try:
            os.makedirs(exports_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create exports directory: {e}")
            sys.exit(1)

    main(dataset_names, algorithm_names, output_directory, export_types, skip_processed, evaluate_only, export_resized)