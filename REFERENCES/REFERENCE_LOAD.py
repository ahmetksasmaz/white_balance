class NUS8Dataset:
    def __init__(self, dataset_root, output_root):
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.camera_names = [
            'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200',
            'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57'
        ]
        self.all_meta_data = []
        
    def setup_directories(self):
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
            
        # Create common output directories
        self.mask_output_dir = os.path.join(self.output_root, 'mask')
        self.img_output_dir = os.path.join(self.output_root, 'images')
        
        if not os.path.exists(self.mask_output_dir):
            os.makedirs(self.mask_output_dir)
        if not os.path.exists(self.img_output_dir):
            os.makedirs(self.img_output_dir)
            
    def process_camera(self, camera_name):
        ground_truth = scipy.io.loadmat(os.path.join(self.dataset_root, camera_name, "ground_truth", f"{camera_name}_gt.mat"))
        filenames = sorted(os.listdir(os.path.join(self.dataset_root, camera_name, "PNG")))
        
        # Get all the metadata
        illuminants = ground_truth['groundtruth_illuminants']
        darkness_level = ground_truth['darkness_level']
        saturation_level = ground_truth['saturation_level']
        cc_coords = ground_truth['CC_coords']
        
        # Normalize illuminants
        illuminants = illuminants / np.linalg.norm(illuminants, axis=1)[..., np.newaxis]
        
        for idx, file in enumerate(filenames):
            print(f"Processing {camera_name} file: {file} ({idx+1}/{len(filenames)})")
            
            # Create individual meta data entry for each image
            meta_entry = {
                'camera_name': camera_name,
                'illuminant': illuminants[idx].tolist(),
                'darkness_level': darkness_level.tolist(),
                'saturation_level': saturation_level.tolist(),
                'cc_coord': cc_coords[idx].tolist(),
                'filename': os.path.basename(file),
            }
            
            # Add to flat list
            self.all_meta_data.append(meta_entry)
            
            # Process and save the image
            self.process_image(camera_name, file, darkness_level, cc_coords, idx)
    
    def process_image(self, camera_name, file, darkness_level, cc_coords, index):
        # Read raw image
        file_path = os.path.join(self.dataset_root, camera_name, "PNG", file)
        raw = np.array(cv2.imread(file_path, -1), dtype='float32')
        raw = np.maximum(raw - darkness_level, [0, 0, 0])

        # Process image
        img = (raw/raw.max() * 65535.0).astype(np.uint16)
        img = np.clip(img, 0, 65535).astype(np.uint16)
        index = int(file.split('_')[1].split('.')[0]) - 1
        
        # Get cc_coords for current image
        cc_coord = cc_coords[index]
        
        # Create mask from cc_coords
        y1, y2, x1, x2 = cc_coord
        mask = np.zeros(raw.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        cv2.imwrite(os.path.join(self.mask_output_dir, file), mask)
        cv2.imwrite(os.path.join(self.img_output_dir, file), img)
    
    def save_metadata(self):
        with open(os.path.join(self.output_root, 'all_cameras_meta.json'), 'w') as f:
            json.dump(self.all_meta_data, f, indent=4)
    
    def process_dataset(self):
        print("Start processing")
        self.setup_directories()
        
        for camera_name in self.camera_names:
            self.process_camera(camera_name)
            
        self.save_metadata()
        print("Finish processing NUS8 dataset")