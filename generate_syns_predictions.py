import os
import sys
import argparse
import numpy as np
import time
import torch
from PIL import Image
from tqdm import tqdm
import zipfile
from io import BytesIO

# Import the models dictionary and prediction function
from models import MODELS, predict_depth

class SynsPatchesAccessor:
    def __init__(self, path_syns_patches_zip, split="val"):
        self.path_syns_patches_zip = path_syns_patches_zip
        self.split = split
        self.split_files = self._load_split_files()

    def _load_split_files(self):
        with zipfile.ZipFile(self.path_syns_patches_zip, 'r') as zip_file:
            split_files_path = f"syns_patches/splits/{self.split}_files.txt"
            if split_files_path not in zip_file.namelist():
                raise FileNotFoundError(f"'{split_files_path}' not found in the ZIP archive of the SYNS-Patches dataset.")
            
            split_files_content = zip_file.read(split_files_path).decode("utf-8")
            return split_files_content.splitlines()

    def __len__(self):
        return len(self.split_files)

    def __iter__(self):
        with zipfile.ZipFile(self.path_syns_patches_zip, 'r') as zip_file:
            for line in self.split_files:
                folder_name, image_name = line.split()
                image_path = f"syns_patches/{folder_name}/images/{image_name}"

                if image_path not in zip_file.namelist():
                    raise FileNotFoundError(f"'{image_path}' not found in the ZIP archive of the SYNS-Patches dataset.")

                image_data = zip_file.read(image_path)
                image = Image.open(BytesIO(image_data))
                image = image.convert('RGB')  # Ensure RGB format
                yield np.array(image), f"{folder_name}/{image_name}"

def generate_predictions(model_id, syns_accessor, output_dir, visualize=True):
    """
    Generate predictions for the SYNS dataset using the specified model.
    
    Args:
        model_id (str): ID of the model to use
        syns_accessor (SynsPatchesAccessor): SYNS dataset accessor
        output_dir (str): Directory to save outputs
        visualize (bool): Whether to generate visualization images
    
    Returns:
        tuple: (predictions array, inference times list)
    """
    print(f"Generating predictions for model: {MODELS[model_id]['name']}")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'npz'), exist_ok=True)
    if visualize:
        vis_dir = os.path.join(output_dir, 'vis', model_id)
        os.makedirs(vis_dir, exist_ok=True)
    
    predictions = []
    inference_times = []
    
    for idx, (img, img_path) in enumerate(tqdm(syns_accessor, desc=f"Processing {model_id}")):
        # Predict depth
        start_time = time.time()
        depth_map = predict_depth(img, model_id)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Store prediction
        predictions.append(depth_map)
        
        # Visualize if requested
        if visualize:
            # Normalize depth for visualization (convert to viridis colormap)
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # Normalize depth between 0-1 for visualization
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            colored_depth = plt.cm.viridis(depth_norm)[:, :, :3]
            colored_depth = (colored_depth * 255).astype(np.uint8)
            
            # Save visualization
            vis_path = os.path.join(vis_dir, f"{idx:04d}_{img_path.replace('/', '_')}.png")
            Image.fromarray(colored_depth).save(vis_path)
    
    # Convert predictions to numpy array
    predictions = np.stack(predictions)
    
    # Save as NPZ file
    npz_path = os.path.join(output_dir, 'npz', f"{model_id}.npz")
    np.savez(npz_path, pred=predictions, pred_type="depth")
    
    # Calculate average inference time
    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time for {model_id}: {avg_inference_time:.4f}s")
    
    return predictions, inference_times

def main():
    parser = argparse.ArgumentParser(description="Generate depth predictions for SYNS-Patches dataset")
    parser.add_argument("--syns_zip", required=True, help="Path to syns_patches.zip dataset")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split to use")
    parser.add_argument("--output_dir", default="submissions", help="Directory to save outputs")
    parser.add_argument("--models", nargs="+", help="Model IDs to evaluate (default: all models)")
    parser.add_argument("--no_vis", action="store_true", help="Skip generating visualization images")
    args = parser.parse_args()
    
    # Check if the SYNS dataset exists
    if not os.path.exists(args.syns_zip):
        print(f"Error: SYNS-Patches zip file not found at {args.syns_zip}")
        return
    
    # Initialize the SYNS dataset accessor
    syns_accessor = SynsPatchesAccessor(args.syns_zip, args.split)
    print(f"Found {len(syns_accessor)} images in {args.split} split")
    
    # Determine which models to evaluate
    if args.models:
        models_to_evaluate = [m for m in args.models if m in MODELS]
        if not models_to_evaluate:
            print("Error: No valid models specified")
            return
    else:
        models_to_evaluate = list(MODELS.keys())
    
    # Generate predictions for each model
    for model_id in models_to_evaluate:
        generate_predictions(model_id, syns_accessor, args.output_dir, not args.no_vis)
    
    print(f"All predictions saved to {args.output_dir}")

if __name__ == "__main__":
    main()