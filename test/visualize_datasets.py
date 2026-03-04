import sys
import os
import random
import cv2 as cv
import numpy as np

# Add parent directory to sys.path to access the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator import DATASET_PROVIDERS
from visuals.log_chrominance_histogram import LogChrominanceHistogram
from visuals.normalized_rgb_histogram import NormalizedRGBHistogram

def resize_and_pad(img, size=300, pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    
    # interpolation method
    if h > size or w > size:
        interp = cv.INTER_AREA
    else:
        interp = cv.INTER_CUBIC
        
    aspect = w/h
    
    # compute new dimensions
    if aspect > 1:
        new_w = size
        new_h = int(np.round(new_w / aspect))
    elif aspect < 1:
        new_h = size
        new_w = int(np.round(new_h * aspect))
    else:
        new_h, new_w = size, size
        
    # resize
    scaled_img = cv.resize(img, (new_w, new_h), interpolation=interp)
    
    # create canvas
    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
    
    # pad to center
    pad_y = (size - new_h) // 2
    pad_x = (size - new_w) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = scaled_img
    
    return canvas

def main():
    cell_size = 300
    margin = 15
    grid_rows, grid_cols = 3, 3
    images_per_row = 3
    bg_color = (40, 40, 40) # dark gray margins
    
    canvas_w = grid_cols * cell_size + (grid_cols + 1) * margin
    canvas_h = grid_rows * cell_size + (grid_rows + 1) * margin

    # Initialize visualizers from the visuals folder
    log_chroma_vis = LogChrominanceHistogram(image_size=cell_size)
    rgb_hist_vis = NormalizedRGBHistogram(image_size=cell_size)
    
    for dataset_name, provider_class in DATASET_PROVIDERS.items():
        print(f"\nProcessing dataset: {dataset_name}")
        try:
            provider = provider_class()
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue
            
        dataset_size = len(provider)
        if dataset_size == 0:
            print(f"Dataset {dataset_name} is empty.")
            continue
            
        actual_num_images = min(images_per_row, dataset_size)
        random_indices = random.sample(range(dataset_size), actual_num_images)
        
        print(f"Visualizing {actual_num_images} images with histograms from {dataset_name}.")
        
        row1_images = [] # Original images
        row2_images = [] # Log-chroma histograms
        row3_images = [] # RGB histograms
        
        for idx in random_indices:
            try:
                data = provider[idx]
                raw_img = data.get_raw_image()
                
                if raw_img is None:
                    print(f"Failed to load image at index {idx}")
                    empty = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                    row1_images.append(empty)
                    row2_images.append(empty)
                    row3_images.append(empty)
                    continue
                
                # 1. Prepare Display Image (Gamma Corrected)
                corrected_img = np.power(np.clip(raw_img, 0, 1), 1.0 / 2.2)
                display_img = (corrected_img * 255.0).clip(0, 255).astype(np.uint8)
                if len(display_img.shape) == 2:
                    display_img = cv.cvtColor(display_img, cv.COLOR_GRAY2BGR)
                row1_images.append(resize_and_pad(display_img, size=cell_size))
                
                # 2. Log Chrominance Histogram (using visualizer class)
                log_chroma_img, _ = log_chroma_vis.visualize(data)
                row2_images.append(cv.resize(log_chroma_img, (cell_size, cell_size)))
                
                # 3. Normalized RGB Histogram (using visualizer class)
                rgb_hist_img, _ = rgb_hist_vis.visualize(data)
                row3_images.append(cv.resize(rgb_hist_img, (cell_size, cell_size)))
                
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                empty = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                row1_images.append(empty)
                row2_images.append(empty)
                row3_images.append(empty)

        # Pad rows if indices < 3
        while len(row1_images) < 3:
            empty = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
            row1_images.append(empty)
            row2_images.append(empty)
            row3_images.append(empty)
            
        # Draw on the main canvas
        grid_canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
        
        all_cells = row1_images + row2_images + row3_images
        
        for i, cell_img in enumerate(all_cells):
            r = i // 3
            c = i % 3
            y_offset = margin + r * (cell_size + margin)
            x_offset = margin + c * (cell_size + margin)
            grid_canvas[y_offset:y_offset+cell_size, x_offset:x_offset+cell_size] = cell_img
        
        window_name = f"{dataset_name} Visualization (3x3)"
        cv.imshow(window_name, grid_canvas)
        
        print("Press any key to show the next dataset. Press ESC to exit.")
        key = cv.waitKey(0)
        cv.destroyWindow(window_name)
        
        if key == 27: # ESC
            break

    cv.destroyAllWindows()
    print("\nFinished viewing datasets.")

if __name__ == "__main__":
    main()
