import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

def normalize_predictions(input_dir, output_dir, model_id, normalization='global'):
    """
    Normalize depth predictions from a model for submission.
    
    Args:
        input_dir (str): Directory containing NPZ prediction files
        output_dir (str): Directory to save normalized predictions
        model_id (str): Model ID to normalize
        normalization (str): Normalization method ('global', 'per_image', 'affine_invariant')
    
    Returns:
        str: Path to normalized predictions NPZ file
    """
    print(f"Normalizing predictions for model: {model_id}")
    
    # Load predictions
    input_path = os.path.join(input_dir, 'npz', f"{model_id}.npz")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Predictions file not found: {input_path}")
    
    data = np.load(input_path)
    predictions = data['pred']
    pred_type = data.get('pred_type', 'depth')
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'normalized'), exist_ok=True)
    vis_dir = os.path.join(output_dir, 'vis_normalized', model_id)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Convert disparity to depth if needed
    if pred_type == 'disparity':
        print("Converting disparity to depth...")
        predictions = 1.0 / np.maximum(predictions, 1e-6)
    
    # Apply normalization
    if normalization == 'global':
        # Global min-max normalization to 0-1
        min_val = np.min(predictions)
        max_val = np.max(predictions)
        norm_predictions = (predictions - min_val) / (max_val - min_val + 1e-8)
        print(f"Global normalization: min={min_val:.4f}, max={max_val:.4f}")
    
    elif normalization == 'per_image':
        # Normalize each image independently
        norm_predictions = np.zeros_like(predictions)
        for i in range(predictions.shape[0]):
            min_val = np.min(predictions[i])
            max_val = np.max(predictions[i])
            norm_predictions[i] = (predictions[i] - min_val) / (max_val - min_val + 1e-8)
        print("Per-image normalization applied")
    
    elif normalization == 'affine_invariant':
        # Affine-invariant normalization (popular for monocular depth)
        norm_predictions = np.zeros_like(predictions)
        for i in range(predictions.shape[0]):
            depth = predictions[i]
            mask_valid = depth > 0
            depth_valid = depth[mask_valid]
            
            # Use 5% and 95% percentiles to avoid outliers
            d_min = np.percentile(depth_valid, 5)
            d_max = np.percentile(depth_valid, 95)
            
            norm_depth = (depth - d_min) / (d_max - d_min + 1e-8)
            norm_depth = np.clip(norm_depth, 0, 1)
            norm_predictions[i] = norm_depth
        print("Affine-invariant normalization applied")
    
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    # Save normalized predictions
    output_path = os.path.join(output_dir, 'normalized', f"{model_id}_{normalization}.npz")
    np.savez(output_path, pred=norm_predictions, normalization=normalization)
    
    # Generate visualizations
    for i in tqdm(range(norm_predictions.shape[0]), desc="Generating visualizations"):
        # Apply colormap
        depth_vis = plt.cm.viridis(norm_predictions[i])[:, :, :3]
        depth_vis = (depth_vis * 255).astype(np.uint8)
        
        # Save visualization
        vis_path = os.path.join(vis_dir, f"{i:04d}.png")
        Image.fromarray(depth_vis).save(vis_path)
    
    print(f"Normalized predictions saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Normalize depth predictions for submission")
    parser.add_argument("--input_dir", default="submissions", help="Directory containing NPZ prediction files")
    parser.add_argument("--output_dir", default="submissions", help="Directory to save normalized predictions")
    parser.add_argument("--models", nargs="+", help="Model IDs to normalize")
    parser.add_argument("--normalization", default="affine_invariant", 
                        choices=["global", "per_image", "affine_invariant"], 
                        help="Normalization method")
    args = parser.parse_args()
    
    # Find available models if not specified
    if not args.models:
        npz_dir = os.path.join(args.input_dir, 'npz')
        if not os.path.exists(npz_dir):
            print(f"Error: NPZ directory not found at {npz_dir}")
            return
        
        available_models = [os.path.splitext(f)[0] for f in os.listdir(npz_dir) if f.endswith('.npz')]
        if not available_models:
            print(f"Error: No NPZ prediction files found in {npz_dir}")
            return
        
        args.models = available_models
    
    # Normalize predictions for each model
    for model_id in args.models:
        try:
            normalize_predictions(args.input_dir, args.output_dir, model_id, args.normalization)
        except Exception as e:
            print(f"Error normalizing predictions for {model_id}: {e}")
    
    print(f"All normalizations complete")

if __name__ == "__main__":
    main()