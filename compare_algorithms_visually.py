import numpy as np
import cv2 as cv
import sys
import os
import argparse
from configuration import *
from datasets.cubepp.cubepp_dataloader import CubePPDataLoader
from datasets.lsmi.lsmi_dataloader import LSMIDataLoader
from helper import *
import tqdm

def main(dataset_names, algorithm_names, output_directory, export_resized, skip_processed):
    # Load database
    postfix = "resized" if export_resized else "full"
    LABEL_HEIGHT = 100
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 3
    for dataset_name in dataset_names:
        dataloader = None
        if dataset_name == "cubepp":
            dataloader = CubePPDataLoader()
        elif dataset_name == "lsmi":
            dataloader = LSMIDataLoader()
        for i in tqdm.tqdm(range(len(dataloader))):
            data = dataloader[i]
            export_path = os.path.join(output_directory, "exports", "comparisons", f"{data.get_image_name()}_{dataset_name}_comparison_{postfix}.png")
            if skip_processed and os.path.exists(export_path):
                continue
            # Check if gt is available
            gt_path = os.path.join(output_directory, "exports", f"{data.get_image_name()}_{dataset_name}_gt_{postfix}.png")
            if not os.path.exists(gt_path):
                continue
            orig_image = data.get_image()
            orig_image = prepare_display(orig_image, correct_gamma=True)
            if export_resized:
                orig_h, orig_w = orig_image.shape[:2]
                new_w = int(orig_w * (512 / orig_h))
                orig_image = cv.resize(orig_image, (new_w, 512))
            gt_image = cv.imread(gt_path)
            h, w, _ = orig_image.shape
            comparison_image = np.zeros((h * 2 + LABEL_HEIGHT, w * (len(algorithm_names)+1), 3), dtype=np.uint8)
            comparison_image[LABEL_HEIGHT:h+LABEL_HEIGHT, 0:w, :] = orig_image
            comparison_image[h+LABEL_HEIGHT:h*2+LABEL_HEIGHT, 0:w, :] = gt_image
            cv.putText(comparison_image, "Input / GT", (10, LABEL_HEIGHT // 2), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
            for algorithm_name in algorithm_names:
                algo_path = os.path.join(output_directory, "exports", f"{data.get_image_name()}_{dataset_name}_{algorithm_name}_algorithm_{postfix}.png")
                if not os.path.exists(algo_path):
                    continue
                heatmap_path = os.path.join(output_directory, "exports", f"{data.get_image_name()}_{dataset_name}_{algorithm_name}_heatmap_{postfix}.png")
                algo_image = cv.imread(algo_path)
                heatmeap_image = cv.imread(heatmap_path)
                comparison_image[LABEL_HEIGHT:h+LABEL_HEIGHT, w * (algorithm_names.index(algorithm_name)+1):w * (algorithm_names.index(algorithm_name)+2), :] = algo_image
                comparison_image[h+LABEL_HEIGHT:h*2+LABEL_HEIGHT, w * (algorithm_names.index(algorithm_name)+1):w * (algorithm_names.index(algorithm_name)+2), :] = heatmeap_image
                cv.putText(comparison_image, f"{algorithm_name}", (w * (algorithm_names.index(algorithm_name)+1) + 10, LABEL_HEIGHT // 2), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
            cv.imwrite(export_path, comparison_image)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White Balance Algorithm Result Visual Comparison")
    parser.add_argument("--datasets", type=str, required=False, default="all", help="List of the name of the datasets to run [cubepp, lsmi], type 'all' for all")
    parser.add_argument("--algorithms", type=str, required=False, default="all", help="List of the algorithms to run [gray_world, max_rgb], type 'all' for all")
    parser.add_argument("--output", type=str, required=True, default="output", help="Path to the output directory")
    parser.add_argument("--export_resized", action="store_true", help="Export resized images (height 512px)")
    parser.add_argument("--skip", action="store_true", help="Skip already processed images with that algorithm (scans the output directory for formatted filenames)")
    args = parser.parse_args()

    dataset_names = args.datasets.split(",")
    algorithm_names = args.algorithms.split(",")
    output_directory = args.output
    export_resized = args.export_resized
    skip_processed = args.skip

    if "all" in dataset_names:
        dataset_names = VALID_DATASETS
    if "all" in algorithm_names:
        algorithm_names = VALID_ALGORITHMS
    
    for dataset in dataset_names:
        if dataset not in VALID_DATASETS:
            print(f"Invalid dataset: {dataset}")
            sys.exit(1)
    for algo in algorithm_names:
        if algo not in VALID_ALGORITHMS:
            print(f"Invalid algorithm: {algo}")
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

    # Try to create exports/comparisons directory in output directory if not exists
    comparison_exports_dir = os.path.join(output_directory, "exports", "comparisons")
    if not os.path.exists(comparison_exports_dir):
        try:
            os.makedirs(comparison_exports_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create exports directory: {e}")
            sys.exit(1)

    main(dataset_names, algorithm_names, output_directory, export_resized, skip_processed)