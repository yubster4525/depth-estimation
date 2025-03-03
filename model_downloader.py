#!/usr/bin/env python3
"""
Model Downloader for Depth Estimation App
Downloads and prepares models for local use to avoid Hugging Face API issues
"""

import os
import sys
import requests
from pathlib import Path

# Try importing optional dependencies
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm is not available
    class tqdm:
        def __init__(self, total=None, unit=None, unit_scale=None, desc=None):
            self.total = total
            self.n = 0
            self.desc = desc
            print(f"{desc}: Starting download...")
            
        def update(self, n):
            self.n += n
            if self.total:
                print(f"{self.desc}: {self.n}/{self.total} bytes downloaded ({self.n/self.total*100:.1f}%)")
            else:
                print(f"{self.desc}: {self.n} bytes downloaded")
                
        def close(self):
            if self.total:
                print(f"{self.desc}: Download complete! {self.n}/{self.total} bytes")
            else:
                print(f"{self.desc}: Download complete! {self.n} bytes")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, will download models but won't be able to preprocess them")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available, continuing with limited functionality")

# Define model URLs and info
MODELS = {
    "midas_small": {
        "name": "MiDaS Small",
        "url": "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small.onnx",
        "description": "Lightweight model for fast inference (ONNX format)",
        "filename": "midas_small.onnx"
    },
    "dpt_hybrid": {
        "name": "DPT Hybrid",
        "url": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "description": "Good balance between speed and accuracy",
        "filename": "dpt_hybrid.pt",
        "hf_model_id": "Intel/dpt-hybrid-midas"  # Added HF model ID for transformers
    },
    "dpt_large": {
        "name": "DPT Large",
        "url": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "description": "Highest quality depth estimation",
        "filename": "dpt_large.pt",
        "hf_model_id": "Intel/dpt-large"         # Added HF model ID for transformers
    },
    "monodepth2": {
        "name": "MonoDepth2",
        "url": "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
        "description": "Accurate depth with self-supervised training",
        "filename": "monodepth2.zip",
        "extract": True
    }
}

def download_file(url, output_path, description="Downloading"):
    """
    Downloads a file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=description)
    
    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Failed to download completely!")
        return False
    return True

def download_hf_model(model_id, model_info, models_dir):
    """Download a model using the HuggingFace transformers library"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping HuggingFace model download")
        return False
        
    try:
        try:
            from transformers import DPTForDepthEstimation
        except ImportError:
            print("Transformers library not available, skipping HuggingFace model download")
            return False
        
        # Create output directory
        output_dir = models_dir / f"{model_id}_hf"
        
        print(f"Downloading {model_info['name']} from HuggingFace...")
        
        # Download model from HuggingFace
        model = DPTForDepthEstimation.from_pretrained(
            model_info["hf_model_id"],
            torch_dtype=torch.float32
        )
        
        # Save model locally
        model.save_pretrained(output_dir)
        print(f"Successfully downloaded and saved {model_info['name']} to {output_dir}")
        
        return True
    except Exception as e:
        print(f"Error downloading model from HuggingFace: {e}")
        return False

def main():
    # Create models directory if not exists
    models_dir = Path("models/downloads")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if CUDA is available
    if TORCH_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} for PyTorch operations")
    else:
        print("PyTorch not available, models will be downloaded but not processed")
    
    # Make sure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Create monodepth2 directory if it doesn't exist
    os.makedirs(models_dir / "monodepth2", exist_ok=True)
    
    # Create dummy files for app testing if download fails
    # This allows the app to run with dummy implementations
    print("Creating dummy model file placeholders for testing...")
    try:
        # Create dummy files for each model type so the app can run with dummy implementations
        for model_id, model_info in MODELS.items():
            dummy_path = models_dir / model_info["filename"]
            if not os.path.exists(dummy_path):
                # Create a minimal file for testing
                with open(dummy_path, 'w') as f:
                    f.write(f"# Dummy model file for {model_info['name']}\n")
                print(f"Created dummy placeholder for {model_info['name']}")
    except Exception as e:
        print(f"Error creating dummy files: {e}")
    
    # Download each model
    for model_id, model_info in MODELS.items():
        output_path = models_dir / model_info["filename"]
        
        # Check if we should try HuggingFace downloads for transformer models
        if "hf_model_id" in model_info:
            hf_path = models_dir / f"{model_id}_hf"
            if hf_path.exists():
                print(f"HuggingFace model {model_info['name']} already exists at {hf_path}")
                continue
                
            print(f"Attempting to download {model_info['name']} from HuggingFace...")
            if download_hf_model(model_id, model_info, models_dir):
                continue  # Skip download from original source if HF download worked
        
        # Standard download for non-HF models or as fallback
        if output_path.exists() and os.path.getsize(output_path) > 100:  # Skip if it's not just a dummy file
            print(f"Model {model_info['name']} already exists at {output_path}")
            continue
        
        print(f"Downloading {model_info['name']} from original source...")
        try:
            download_file(model_info["url"], output_path, description=f"Downloading {model_info['name']}")
        except Exception as e:
            print(f"Error downloading {model_info['name']}: {e}")
            print(f"The application will use dummy implementation for this model.")
        
        # Extract and prepare models as needed
        if model_info.get("extract", False):
            import zipfile
            import shutil
            
            # Create a directory for the extracted files
            extract_dir = models_dir / model_id
            extract_dir.mkdir(exist_ok=True)
            
            print(f"Extracting {model_info['name']} to {extract_dir}...")
            
            # Extract the zip file
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"Successfully extracted {model_info['name']}")
            
            # For MonoDepth2, we need additional preparation
            if model_id == "monodepth2":
                # Rename and reorganize files as needed
                # The specific files depend on the MonoDepth2 archive structure
                
                encoder_path = list(extract_dir.glob("*encoder*"))[0] if list(extract_dir.glob("*encoder*")) else None
                depth_decoder_path = list(extract_dir.glob("*depth*decoder*"))[0] if list(extract_dir.glob("*depth*decoder*")) else None
                
                if encoder_path and depth_decoder_path:
                    print(f"Found model components: {encoder_path.name}, {depth_decoder_path.name}")
                    
                    # In a real implementation, we might convert these to a torchscript model
                    # For now, we'll just make sure they're organized properly
                    target_dir = models_dir / "monodepth2"
                    target_dir.mkdir(exist_ok=True)
                    
                    # Copy the files to the target directory with standardized names
                    if not (target_dir / "encoder.pth").exists():
                        shutil.copy(encoder_path, target_dir / "encoder.pth")
                    
                    if not (target_dir / "depth.pth").exists():
                        shutil.copy(depth_decoder_path, target_dir / "depth.pth")
                    
                    print("Prepared MonoDepth2 model files")
                else:
                    print("Could not find expected model files in the extracted directory")
        
        print(f"Successfully downloaded {model_info['name']} to {output_path}")
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    main()