import sys
import os
import cv2 as cv
import numpy as np

# Add parent directory to sys.path to access the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator import DATASET_PROVIDERS
from helper import *

def main():
    image_index = 0
    if len(sys.argv) == 2:
        try:
            image_index = int(sys.argv[1])
        except ValueError:
            print("Usage: python test_dataproviders.py [image_index]")
            sys.exit(1)

    print(f"Testing dataproviders with image_index={image_index}\n")

    for dataset_name, provider_class in DATASET_PROVIDERS.items():
        print(f"{'='*40}")
        print(f"Testing Dataset: {dataset_name}")
        print(f"{'='*40}")
        
        try:
            provider = provider_class()
            dataset_len = len(provider)
            print(f"Total Images in {dataset_name}: {dataset_len}")
            
            if dataset_len == 0:
                print("Dataset is empty.\n")
                continue

            if image_index < dataset_len:
                test_data = provider[image_index]
                
                print("Image Name:", test_data.get_image_name())
                print("Image Dimensions:", test_data.get_image_dimensions())
                print("GT Info:", test_data.get_illuminants())
                
                # Check for illuminant map (used in LSMI, etc.)
                illuminant_map = test_data.get_illuminant_map()
                if illuminant_map is not None:
                    try:
                        print("Illuminant Map Shape:", illuminant_map.shape)
                    except AttributeError:
                        # Fallback for unexpected type
                        print("Illuminant Map Type:", type(illuminant_map))
                        
            else:
                print(f"Requested image_index {image_index} is out of bounds (Dataset size: {dataset_len}).")
                
        except Exception as e:
            print(f"Error initializing or reading from {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            
        print("\n")

if __name__ == "__main__":
    main()
