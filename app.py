import os
import io
import base64
import numpy as np
import json
import glob
from collections import defaultdict

# Try to import cv2, but fall back to PIL if necessary
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available, using PIL for image processing")

from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import time
import torch
import torch.nn.functional as F

# Import your models, model dictionary, and the predict_depth function
from models import MODELS, predict_depth

from pathlib import Path

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
GROUND_TRUTH_FOLDER = os.path.join('static', 'ground_truth')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(GROUND_TRUTH_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['GROUND_TRUTH_FOLDER'] = GROUND_TRUTH_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload


# -------------------------------
# Utility: Ensure model directories
# -------------------------------
def ensure_model_dirs():
    """
    Create any required subdirectories to hold downloaded model files.
    Adjust this logic if your local structure differs.
    """
    os.makedirs('models/downloads', exist_ok=True)

    # Create subdirectories for each model if needed
    for model_id, model_info in MODELS.items():
        base_dir = os.path.join('models/downloads', model_id)
        os.makedirs(base_dir, exist_ok=True)


# -------------------------------
# Utility: Check if all models are present
# -------------------------------
def check_models():
    """
    Verify that all required model files exist at the specified paths.
    Returns:
        models_exist: bool
        missing_models: list of model_ids not found
    """
    models_exist = True
    missing_models = []

    ensure_model_dirs()

    for model_id, model_info in MODELS.items():
        path = model_info["path"]
        if not os.path.exists(path):
            models_exist = False
            missing_models.append(model_id)

    return models_exist, missing_models


# -------------------------------
# Core depth estimation routine
# -------------------------------
def estimate_depth(image_path, model_id):
    """
    Perform depth estimation on an image using the specified model.
    
    Args:
        image_path (str): Path to the input image file.
        model_id (str): Identifier for which model to use from MODELS dict.
    
    Returns:
        combined (PIL.Image): Side-by-side RGB + depth colormap.
        depth_image (PIL.Image): Depth colormap alone.
        inference_time (float): Time (seconds) to run inference.
        depth_min (float): Minimum depth value in predicted map.
        depth_max (float): Maximum depth value in predicted map.
    """
    # Load image
    if CV2_AVAILABLE:
        # Using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        # Convert from BGR to RGB
        orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
    else:
        # Using PIL
        pil_image = Image.open(image_path).convert('RGB')
        orig_rgb = np.array(pil_image)
        height, width = orig_rgb.shape[:2]

    # Start timing
    start_time = time.time()

    # -----------------------
    # Actual depth prediction
    # -----------------------
    depth_map = predict_depth(orig_rgb, model_id)

    # End timing
    inference_time = time.time() - start_time

    # Generate a color-mapped depth image using matplotlib
    colored_depth = plt.cm.viridis(depth_map)[:, :, :3]  # shape: (H, W, 3)
    colored_depth = (colored_depth * 255).astype(np.uint8)

    # Convert to PIL
    rgb_image_pil = Image.fromarray(orig_rgb)
    depth_image_pil = Image.fromarray(colored_depth)

    # Side-by-side comparison (original + depth)
    combined = Image.new('RGB', (width * 2, height))
    combined.paste(rgb_image_pil, (0, 0))
    combined.paste(depth_image_pil, (width, 0))

    # Compute min/max of depth
    depth_min = float(depth_map.min())
    depth_max = float(depth_map.max())

    return combined, depth_image_pil, inference_time, depth_min, depth_max


# -------------------------------
# Flask Routes
# -------------------------------
@app.route('/')
def index():
    """
    Main page. Checks if models are present; if not, shows the download page.
    Otherwise, renders index.html with the available models.
    """
    from datetime import datetime
    
    models_exist, missing_models = check_models()

    if not models_exist:
        # If any model files are missing, show the download instructions page
        return render_template('download.html', missing_models=missing_models)

    # Prepare model info for the template
    template_models = {}
    for model_id, model_info in MODELS.items():
        template_models[model_id] = {
            'name': model_info['name'],
            'description': model_info['description']
        }

    # Set context for template
    context = {
        'models': template_models,
        'active_page': 'home'
    }
    
    return render_template('index.html', **context)

@app.route('/results')
def results_page():
    """
    Results page. Shows NPZ result folders and allows visualization.
    """
    from datetime import datetime
    
    models_exist, missing_models = check_models()

    if not models_exist:
        # If any model files are missing, show the download instructions page
        return render_template('download.html', missing_models=missing_models)
    
    # Get folders in the npz directory
    npz_dir = os.path.join(app.config['RESULT_FOLDER'], 'npz')
    folders = []
    
    if os.path.exists(npz_dir):
        for folder_name in os.listdir(npz_dir):
            folder_path = os.path.join(npz_dir, folder_name)
            
            if os.path.isdir(folder_path):
                # Count files
                file_count = len([f for f in os.listdir(folder_path) if f.endswith('.npz')])
                
                # Get folder size
                folder_size = 0
                for dirpath, dirnames, filenames in os.walk(folder_path):
                    for filename in filenames:
                        file_path = os.path.join(dirpath, filename)
                        folder_size += os.path.getsize(file_path)
                
                # Format size
                if folder_size > 1024 * 1024 * 1024:
                    size_str = f"{folder_size / (1024 * 1024 * 1024):.2f} GB"
                elif folder_size > 1024 * 1024:
                    size_str = f"{folder_size / (1024 * 1024):.2f} MB"
                else:
                    size_str = f"{folder_size / 1024:.2f} KB"
                
                # Get creation time
                created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getctime(folder_path)))
                
                folders.append({
                    'name': folder_name,
                    'path': folder_path,
                    'file_count': file_count,
                    'size': size_str,
                    'created': created
                })
    
    # Sort folders by creation time (newest first)
    folders.sort(key=lambda x: x['name'], reverse=True)
    
    context = {
        'folders': folders,
        'active_page': 'results'
    }
    
    return render_template('results.html', **context)


@app.route('/dataset')
def dataset_page():
    """
    Dataset evaluation page. Allows batch processing of datasets.
    """
    models_exist, missing_models = check_models()

    if not models_exist:
        # If any model files are missing, show the download instructions page
        return render_template('download.html', missing_models=missing_models)

    # Add custom models to MODELS dictionary if they exist
    # This will make trained MDEC UNet models available for evaluation
    custom_models_dir = os.path.join(os.getcwd(), 'custom_models', 'trained')
    if os.path.exists(custom_models_dir):
        for model_id in os.listdir(custom_models_dir):
            model_path = os.path.join(custom_models_dir, model_id)
            
            # Check if it has required files
            if (os.path.isdir(model_path) and 
                os.path.exists(os.path.join(model_path, 'model.pth')) and
                os.path.exists(os.path.join(model_path, 'info.json'))):
                
                # Load model info
                with open(os.path.join(model_path, 'info.json'), 'r') as f:
                    model_info = json.load(f)
                
                # Add model to MODELS dictionary if it's not already there
                custom_model_id = f"custom_{model_id}"
                if custom_model_id not in MODELS:
                    MODELS[custom_model_id] = {
                        'name': model_info.get('name', f'Custom Model {model_id}'),
                        'path': os.path.join(model_path, 'model.pth'),
                        'type': model_info.get('type', 'unet') if 'mdec' not in model_info.get('type', '') else 'mdec_unet',
                        'input_size': model_info.get('input_size', (256, 256)),
                        'description': model_info.get('description', 'Custom trained depth estimation model'),
                        'is_custom': True
                    }

    # Prepare model info for the template
    template_models = {}
    for model_id, model_info in MODELS.items():
        template_models[model_id] = {
            'name': model_info['name'],
            'description': model_info['description'],
            'type': model_info['type']  # Include model type to identify MDEC models
        }
    
    # Get available datasets
    available_datasets = []
    custom_datasets_dir = os.path.join(os.getcwd(), 'custom_datasets')
    
    if os.path.exists(custom_datasets_dir):
        for dataset_name in os.listdir(custom_datasets_dir):
            dataset_path = os.path.join(custom_datasets_dir, dataset_name)
            
            # Skip non-directories
            if not os.path.isdir(dataset_path):
                continue
                
            # Count images - look for direct images or nested folders
            image_count = 0
            
            # Check if this is a KITTI dataset
            is_kitti = False
            if os.path.exists(os.path.join(dataset_path, 'splits')):
                # Count KITTI images in split files
                for split_name in os.listdir(os.path.join(dataset_path, 'splits')):
                    split_dir = os.path.join(dataset_path, 'splits', split_name)
                    if os.path.isdir(split_dir):
                        for mode in ['train', 'val', 'test']:
                            split_file = os.path.join(split_dir, f'{mode}_files.txt')
                            if os.path.exists(split_file):
                                with open(split_file, 'r') as f:
                                    image_count += len(f.readlines())
                
                if image_count > 0:
                    is_kitti = True
            
            # If not KITTI, check other dataset formats
            if not is_kitti:
                # First check if there are direct images in an 'images' subfolder
                if os.path.exists(os.path.join(dataset_path, 'images')):
                    direct_image_dir = os.path.join(dataset_path, 'images')
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                        image_count += len(glob.glob(os.path.join(direct_image_dir, ext)))
                
                # For datasets like SYNS-Patches with nested structure
                if image_count == 0:
                    # Look for nested image directories
                    for subdir in glob.glob(os.path.join(dataset_path, '*')):
                        if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, 'images')):
                            nested_image_dir = os.path.join(subdir, 'images')
                            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                                image_count += len(glob.glob(os.path.join(nested_image_dir, ext)))
                
                # Fallback to direct images in the dataset folder
                if image_count == 0:
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                        image_count += len(glob.glob(os.path.join(dataset_path, ext)))
            
            if image_count > 0:
                # Check if it has split files
                has_splits = os.path.exists(os.path.join(dataset_path, 'splits'))
                
                available_datasets.append({
                    'name': dataset_name,
                    'path': dataset_path,
                    'image_count': image_count,
                    'has_splits': has_splits,
                    'is_kitti': is_kitti
                })

    context = {
        'models': template_models, 
        'available_datasets': available_datasets,
        'active_page': 'dataset'
    }

    return render_template('dataset.html', **context)


@app.route('/training')
def training_page():
    """
    Training page. Allows training custom models on datasets.
    """
    models_exist, missing_models = check_models()

    if not models_exist:
        # If any model files are missing, show the download instructions page
        return render_template('download.html', missing_models=missing_models)
    
    # Get available datasets
    datasets = []
    custom_datasets_dir = os.path.join(os.getcwd(), 'custom_datasets')
    
    if os.path.exists(custom_datasets_dir):
        for dataset_name in os.listdir(custom_datasets_dir):
            dataset_path = os.path.join(custom_datasets_dir, dataset_name)
            
            # Check if it's a directory and has required structure
            if (os.path.isdir(dataset_path) and 
                os.path.exists(os.path.join(dataset_path, 'images')) and 
                os.path.exists(os.path.join(dataset_path, 'depths'))):
                
                # Count images
                image_count = len(os.listdir(os.path.join(dataset_path, 'images')))
                
                datasets.append({
                    'name': dataset_name,
                    'path': dataset_path,
                    'image_count': image_count
                })
    
    # Get custom trained models
    custom_models = []
    custom_models_dir = os.path.join(os.getcwd(), 'custom_models', 'trained')
    
    if os.path.exists(custom_models_dir):
        for model_dir in os.listdir(custom_models_dir):
            model_path = os.path.join(custom_models_dir, model_dir)
            
            # Check if it has required files
            if (os.path.isdir(model_path) and 
                os.path.exists(os.path.join(model_path, 'model.pth')) and
                os.path.exists(os.path.join(model_path, 'info.json'))):
                
                # Load model info
                with open(os.path.join(model_path, 'info.json'), 'r') as f:
                    model_info = json.load(f)
                
                # Add model ID
                model_info['id'] = model_dir
                
                custom_models.append(model_info)
    
    context = {
        'datasets': datasets,
        'custom_models': custom_models,
        'active_page': 'training'
    }
    
    return render_template('training.html', **context)


@app.route('/static/<path:filename>')
def serve_static(filename):
    """
    Serve static files from /static directory.
    """
    return send_from_directory('static', filename)


@app.route('/process', methods=['POST'])
def process_image():
    """
    Handles the image upload and depth estimation request.
    Expects form data with 'file' as the image file and 'model' as the model_id.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'})

    uploaded_file = request.files['file']
    model_id = request.form.get('model', 'dpt_hybrid')

    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    if uploaded_file:
        # Save the uploaded image
        orig_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
        uploaded_file.save(orig_filename)

        try:
            # Perform depth estimation
            comparison_img, depth_img, inference_time, depth_min, depth_max = estimate_depth(
                orig_filename, model_id
            )

            # Save results (depth and comparison)
            result_depth_filename = os.path.join(app.config['RESULT_FOLDER'], f'depth_{model_id}.jpg')
            result_comparison_filename = os.path.join(app.config['RESULT_FOLDER'], f'comparison_{model_id}.jpg')

            depth_img.save(result_depth_filename)
            comparison_img.save(result_comparison_filename)

            # Build JSON response
            return jsonify({
                'success': True,
                'original': os.path.join('static', 'uploads', 'input.jpg'),
                'depth_map': os.path.join('static', 'results', f'depth_{model_id}.jpg'),
                'comparison': os.path.join('static', 'results', f'comparison_{model_id}.jpg'),
                'inference_time': f"{inference_time:.2f}",
                'model_used': MODELS[model_id]['name'],
                'depth_min': f"{depth_min:.2f}",
                'depth_max': f"{depth_max:.2f}"
            })

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': str(e)})

    return jsonify({'error': 'Unknown error during processing'})


@app.route('/evaluate-all-models', methods=['GET'])
def evaluate_models():
    """
    Evaluate all models in MODELS dictionary using the last uploaded image.
    Returns a JSON with each model's inference time and the saved depth map path.
    """
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
    if not os.path.exists(input_path):
        return jsonify({'error': 'No image uploaded yet'})

    results = []

    for model_id in MODELS:
        try:
            # Depth estimation for each model
            _, depth_img, inference_time, depth_min, depth_max = estimate_depth(
                input_path, model_id
            )
            # Save the depth map
            depth_output_path = os.path.join(app.config['RESULT_FOLDER'], f'depth_{model_id}.jpg')
            depth_img.save(depth_output_path)

            results.append({
                'model_id': model_id,
                'model_name': MODELS[model_id]['name'],
                'inference_time': f"{inference_time:.2f}",
                'depth_map': os.path.join('static', 'results', f'depth_{model_id}.jpg'),
                'depth_min': f"{depth_min:.2f}",
                'depth_max': f"{depth_max:.2f}"
            })
        except Exception as e:
            results.append({
                'model_id': model_id,
                'model_name': MODELS[model_id]['name'],
                'error': str(e)
            })

    return jsonify({'results': results})


@app.route('/download-models', methods=['GET'])
def download_models_page():
    """
    Simple route to instruct the user on downloading models if they are missing.
    """
    return render_template('download.html')


# -------------------------------
# Ground Truth Evaluation Functions
# -------------------------------
def load_ground_truth(ground_truth_path):
    """
    Load ground truth depth map from an image file.
    
    Args:
        ground_truth_path (str): Path to the ground truth depth map image.
        
    Returns:
        numpy.ndarray: A depth map array with depth values.
    """
    # Check if the file exists
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
    # Try to load the ground truth depth map
    try:
        if CV2_AVAILABLE:
            # OpenCV can load 16-bit depth maps properly
            depth_img = cv2.imread(ground_truth_path, cv2.IMREAD_ANYDEPTH)
            if depth_img is None:
                # Fallback to 8-bit if 16-bit fails
                depth_img = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        else:
            # PIL can handle various formats
            pil_img = Image.open(ground_truth_path)
            depth_img = np.array(pil_img)
            
        # Normalize depth if needed
        if depth_img.dtype == np.uint16:
            # 16-bit depth map, normalize to 0-1 range
            depth_img = depth_img.astype(np.float32) / 65535.0
        elif depth_img.dtype == np.uint8:
            # 8-bit depth map, normalize to 0-1 range
            depth_img = depth_img.astype(np.float32) / 255.0
        
        return depth_img
    except Exception as e:
        raise ValueError(f"Error loading ground truth depth map: {str(e)}")


def create_depth_colormap(depth_map):
    """
    Create a colorized visualization of a depth map.
    
    Args:
        depth_map (numpy.ndarray): The depth map to visualize.
        
    Returns:
        PIL.Image: A colorized visualization of the depth map.
    """
    # Normalize depth to 0-1 range for visualization
    if depth_map.min() != depth_map.max():
        normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    else:
        normalized_depth = depth_map
    
    # Apply colormap
    colored_depth = plt.cm.viridis(normalized_depth)[:, :, :3]  # shape: (H, W, 3)
    colored_depth = (colored_depth * 255).astype(np.uint8)
    
    # Convert to PIL image
    return Image.fromarray(colored_depth)


def extract_edges(depth, preprocess='log', sigma=1, mask=None, use_canny=True):
    """
    Detect edges in a depth map.
    
    Args:
        depth (numpy.ndarray): Depth map to extract edges from.
        preprocess (str): Preprocessing method ('log', 'inv', 'none').
        sigma (int): Gaussian blurring sigma.
        mask (numpy.ndarray, optional): Mask of valid pixels.
        use_canny (bool): If True, use Canny edge detection, otherwise use Sobel.
        
    Returns:
        numpy.ndarray: Binary edge map.
    """
    from skimage.feature import canny
    
    depth = depth.squeeze()
    
    # Preprocess depth map
    if preprocess == 'log':
        # Log transform
        depth = np.log(depth.clip(min=1e-6))
    elif preprocess == 'inv':
        # Inverse transform (disparity)
        depth = 1.0 / depth.clip(min=1e-6)
        depth -= depth.min()  # Normalize disp to emphasize edges
        depth /= depth.max()
    
    # Detect edges
    if use_canny:
        edges = canny(depth, sigma=sigma, mask=mask)
    else:
        # Sobel edge detection
        depth = cv2.GaussianBlur(depth, (3, 3), sigmaX=sigma, sigmaY=sigma)
        dx = cv2.Sobel(src=depth, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        dy = cv2.Sobel(src=depth, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        
        edges = np.sqrt(dx**2 + dy**2)
        edges = edges > edges.mean()
        if mask is not None:
            edges *= mask
    
    return edges

def metrics_edge(pred, target, mask=None):
    """
    Compute edge-based metrics for depth evaluation.
    
    Args:
        pred (numpy.ndarray): Predicted depth map.
        target (numpy.ndarray): Ground truth depth map.
        mask (numpy.ndarray, optional): Mask of valid pixels.
        
    Returns:
        dict: Dictionary of metrics.
    """
    from scipy import ndimage
    
    # Default threshold for distance transform
    th_edges = 10
    
    # Extract edges from ground truth and prediction
    gt_edges = extract_edges(target, preprocess='log', sigma=1, mask=mask)
    pred_edges = extract_edges(pred, preprocess='log', sigma=1, mask=mask)
    
    # Distance transforms
    D_target = ndimage.distance_transform_edt(1 - gt_edges)  # Distance to ground truth edges
    D_pred = ndimage.distance_transform_edt(1 - pred_edges)  # Distance to predicted edges
    
    # Compute precision, recall, and F-score
    # Precision: How many predicted edges are close to ground truth edges
    close_to_gt = (D_target < th_edges)
    precision = np.sum(pred_edges & close_to_gt) / (np.sum(pred_edges) + 1e-6)
    
    # Recall: How many ground truth edges have a predicted edge nearby
    close_to_pred = (D_pred < th_edges)
    recall = np.sum(gt_edges & close_to_pred) / (np.sum(gt_edges) + 1e-6)
    
    # F-score
    f_score = 2 * precision * recall / (precision + recall + 1e-6)
    
    # Edge accuracy and completeness
    edge_acc = D_target[pred_edges].mean() if pred_edges.sum() else th_edges  # Distance from pred to target
    edge_comp = D_pred[gt_edges].mean() if gt_edges.sum() else th_edges  # Distance from target to pred
    
    return {
        'F-Score': float(f_score),
        'Precision': float(precision),
        'Recall': float(recall), 
        'EdgeAcc': float(edge_acc),
        'EdgeComp': float(edge_comp)
    }

def metrics_eigen(pred, target, mask=None):
    """
    Compute Eigen depth evaluation metrics.
    From: Depth Map Prediction from a Single Image using a Multi-Scale Deep Network (Eigen et al., 2014)
    
    Args:
        pred (numpy.ndarray): Predicted depth map.
        target (numpy.ndarray): Ground truth depth map.
        mask (numpy.ndarray, optional): Mask of valid pixels. If None, all pixels are used.
        
    Returns:
        dict: Dictionary of metrics.
    """
    if mask is None:
        # Use all pixels by default
        mask = np.ones_like(pred, dtype=bool)
    
    # Apply mask and flatten arrays
    pred = pred[mask]
    target = target[mask]
    
    # Ensure pred and target have values
    if len(pred) == 0 or len(target) == 0:
        return {
            'AbsRel': float('nan'),
            'SqRel': float('nan'),
            'RMSE': float('nan'),
            'LogRMSE': float('nan'),
            'delta1': float('nan'),
            'delta2': float('nan'),
            'delta3': float('nan'),
        }
    
    # Calculate metrics
    thresh = np.maximum((target / pred), (pred / target))
    
    abs_rel = np.mean(np.abs(pred - target) / target)
    sq_rel = np.mean(((pred - target) ** 2) / target)
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    log_rmse = np.sqrt(np.mean((np.log(pred) - np.log(target)) ** 2))
    
    a1 = np.mean(thresh < 1.25)
    a2 = np.mean(thresh < 1.25 ** 2)
    a3 = np.mean(thresh < 1.25 ** 3)
    
    return {
        'AbsRel': float(abs_rel),
        'SqRel': float(sq_rel),
        'RMSE': float(rmse),
        'LogRMSE': float(log_rmse),
        'delta1': float(a1),
        'delta2': float(a2),
        'delta3': float(a3),
    }


def align_depths(pred, target, mask=None):
    """
    Align predicted depth to ground truth using least squares.
    This function scales and shifts the predicted depth map to match the ground truth.
    
    Args:
        pred (numpy.ndarray): Predicted depth map.
        target (numpy.ndarray): Ground truth depth map.
        mask (numpy.ndarray, optional): Mask of valid pixels. If None, all pixels are used.
        
    Returns:
        numpy.ndarray: Aligned predicted depth map.
    """
    if mask is None:
        # Use all pixels by default
        mask = np.ones_like(pred, dtype=bool)
    
    # Apply mask and reshape for linear regression
    pred_masked = pred[mask].reshape(-1, 1)
    target_masked = target[mask].reshape(-1, 1)
    
    # Add a column of ones for the bias term
    A = np.concatenate([pred_masked, np.ones_like(pred_masked)], axis=1)
    
    # Solve the least squares problem
    x, _, _, _ = np.linalg.lstsq(A, target_masked, rcond=None)
    
    # Extract scale and shift
    scale, shift = x.flatten()
    
    # Apply scale and shift to prediction
    aligned_pred = scale * pred + shift
    
    return aligned_pred


def evaluate_depth_prediction(pred_depth, gt_depth, eval_mode='mono', metrics_types=None):
    """
    Evaluate a depth prediction against ground truth.
    
    Args:
        pred_depth (numpy.ndarray): Predicted depth map.
        gt_depth (numpy.ndarray): Ground truth depth map.
        eval_mode (str): Evaluation mode ('mono' or 'stereo').
            - 'mono': Scale-invariant evaluation (align pred to gt).
            - 'stereo': Fixed-scale evaluation.
        metrics_types (list): List of metric types to compute ('eigen', 'edge').
            
    Returns:
        dict: Dictionary of metrics.
    """
    # Default metrics types if not specified
    if metrics_types is None:
        metrics_types = ['eigen']
    
    # Ensure same shape
    if pred_depth.shape != gt_depth.shape:
        # Resize prediction to match ground truth
        if torch.is_tensor(pred_depth):
            pred_depth = pred_depth.cpu().numpy()
        
        h, w = gt_depth.shape
        if CV2_AVAILABLE:
            pred_depth = cv2.resize(pred_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Use PIL
            pred_depth_pil = Image.fromarray(pred_depth)
            pred_depth_pil = pred_depth_pil.resize((w, h), Image.BILINEAR)
            pred_depth = np.array(pred_depth_pil)
    
    # Create mask for valid pixels (non-zero in ground truth)
    mask = gt_depth > 0
    
    # Skip if no valid pixels
    if not np.any(mask):
        return None
    
    # Align depths if using mono (scale-invariant) evaluation
    if eval_mode == 'mono':
        pred_depth = align_depths(pred_depth, gt_depth, mask)
    
    # Compute all requested metrics
    all_metrics = {}
    
    # Standard Eigen metrics
    if 'eigen' in metrics_types:
        eigen_metrics = metrics_eigen(pred_depth, gt_depth, mask)
        all_metrics.update(eigen_metrics)
    
    # Edge-based metrics
    if 'edge' in metrics_types:
        edge_metrics = metrics_edge(pred_depth, gt_depth, mask)
        all_metrics.update(edge_metrics)
    
    return all_metrics


@app.route('/evaluate', methods=['GET'])
def evaluate_page():
    """
    Render the ground truth evaluation page.
    """
    return render_template('evaluate.html', active_page='evaluate')


@app.route('/evaluate-ground-truth', methods=['POST'])
def evaluate_ground_truth():
    """
    Handle ground truth evaluation request.
    Expects form data with 'image' and 'ground_truth' files.
    """
    if 'image' not in request.files or 'ground_truth' not in request.files:
        return jsonify({'error': 'Missing image or ground truth file'})
    
    # Get uploaded files
    image_file = request.files['image']
    gt_file = request.files['ground_truth']
    
    # Get evaluation mode and metrics
    eval_mode = request.form.get('eval_mode', 'mono')
    selected_metrics = request.form.getlist('metrics')
    
    # Default to Eigen metrics if none selected
    if not selected_metrics:
        selected_metrics = ['eigen']
    
    if image_file.filename == '' or gt_file.filename == '':
        return jsonify({'error': 'No selected files'})
    
    try:
        # Save uploaded files
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'eval_input.jpg')
        gt_path = os.path.join(app.config['GROUND_TRUTH_FOLDER'], 'ground_truth.png')
        
        image_file.save(image_path)
        gt_file.save(gt_path)
        
        # Load ground truth depth map
        gt_depth = load_ground_truth(gt_path)
        print(f"Ground truth loaded successfully. Shape: {gt_depth.shape}, Range: [{gt_depth.min()}, {gt_depth.max()}]")
        
        # Create visualization for ground truth
        gt_viz = create_depth_colormap(gt_depth)
        gt_viz_path = os.path.join(app.config['GROUND_TRUTH_FOLDER'], 'ground_truth_viz.jpg')
        gt_viz.save(gt_viz_path)
        
        # Check if edge metrics were requested
        include_edge_metrics = 'edge' in selected_metrics
        metrics_types = selected_metrics
        
        # If edge metrics are requested, also create edge visualization for ground truth
        if include_edge_metrics:
            gt_edges = extract_edges(gt_depth, preprocess='log', sigma=1)
            gt_edges_viz = Image.fromarray((gt_edges * 255).astype(np.uint8))
            gt_edges_path = os.path.join(app.config['GROUND_TRUTH_FOLDER'], 'ground_truth_edges.jpg')
            gt_edges_viz.save(gt_edges_path)
        
        # Process with all models
        results = []
        
        # Add custom models from mdec_unet if they exist
        # This will make trained MDEC UNet models available for evaluation
        custom_models_dir = os.path.join(os.getcwd(), 'custom_models', 'trained')
        if os.path.exists(custom_models_dir):
            for model_id in os.listdir(custom_models_dir):
                model_path = os.path.join(custom_models_dir, model_id)
                
                # Check if it has required files
                if (os.path.isdir(model_path) and 
                    os.path.exists(os.path.join(model_path, 'model.pth')) and
                    os.path.exists(os.path.join(model_path, 'info.json'))):
                    
                    # Load model info
                    with open(os.path.join(model_path, 'info.json'), 'r') as f:
                        model_info = json.load(f)
                    
                    # Add model to MODELS dictionary if it's not already there
                    custom_model_id = f"custom_{model_id}"
                    if custom_model_id not in MODELS:
                        MODELS[custom_model_id] = {
                            'name': model_info.get('name', f'Custom Model {model_id}'),
                            'path': os.path.join(model_path, 'model.pth'),
                            'type': model_info.get('type', 'unet'),
                            'input_size': model_info.get('input_size', (256, 256)),
                            'description': model_info.get('description', 'Custom trained depth estimation model'),
                            'is_custom': True
                        }
        
        for model_id in MODELS:
            try:
                # Estimate depth
                comparison_img, depth_img, inference_time, depth_min, depth_max = estimate_depth(
                    image_path, model_id
                )
                
                # Save results
                depth_output_path = os.path.join(app.config['RESULT_FOLDER'], f'depth_{model_id}.jpg')
                depth_img.save(depth_output_path)
                
                # Get depth map as numpy array for evaluation
                if CV2_AVAILABLE:
                    img = cv2.imread(image_path)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    rgb_img = np.array(Image.open(image_path).convert('RGB'))
                
                # Get model prediction as raw depth map
                pred_depth = predict_depth(rgb_img, model_id)
                
                # Create edge visualization if requested
                if include_edge_metrics:
                    pred_edges = extract_edges(pred_depth, preprocess='log', sigma=1)
                    pred_edges_viz = Image.fromarray((pred_edges * 255).astype(np.uint8))
                    pred_edges_path = os.path.join(app.config['RESULT_FOLDER'], f'edges_{model_id}.jpg')
                    pred_edges_viz.save(pred_edges_path)
                
                # Compute metrics with all requested metric types
                metrics = evaluate_depth_prediction(pred_depth, gt_depth, eval_mode, metrics_types)
                
                print(f"Model {model_id} metrics: {metrics}")
                
                result = {
                    'model_id': model_id,
                    'model_name': MODELS[model_id]['name'],
                    'inference_time': f"{inference_time:.2f}",
                    'depth_map': os.path.join('static', 'results', f'depth_{model_id}.jpg'),
                    'depth_min': f"{depth_min:.2f}",
                    'depth_max': f"{depth_max:.2f}",
                    'metrics': metrics
                }
                
                # Add edge visualization path if available
                if include_edge_metrics:
                    result['edge_map'] = os.path.join('static', 'results', f'edges_{model_id}.jpg')
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating model {model_id}: {str(e)}")
                results.append({
                    'model_id': model_id,
                    'model_name': MODELS[model_id]['name'],
                    'error': str(e)
                })
        
        # Prepare response with all visualizations
        response = {
            'success': True,
            'original': os.path.join('static', 'uploads', 'eval_input.jpg'),
            'ground_truth_viz': os.path.join('static', 'ground_truth', 'ground_truth_viz.jpg'),
            'results': results
        }
        
        # Add edge visualization if available
        if include_edge_metrics:
            response['ground_truth_edges'] = os.path.join('static', 'ground_truth', 'ground_truth_edges.jpg')
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in ground truth evaluation: {str(e)}")
        return jsonify({'error': str(e)})


# -------------------------------
# Dataset Processing Routes
# -------------------------------
import threading
import queue
import time
import tempfile
import shutil
import os.path as osp

# Global variables for dataset processing
dataset_queue = queue.Queue()
dataset_status = {
    'status': 'idle',
    'current': 0,
    'total': 0,
    'model_name': '',
    'results': []
}
dataset_thread = None
dataset_temp_dir = None

def process_dataset_worker():
    """
    Worker thread to process datasets in the background
    """
    global dataset_status, dataset_temp_dir
    
    try:
        # Get dataset parameters from queue
        params = dataset_queue.get(block=False)
        
        # Create new status dictionary to avoid shared reference issues
        dataset_status.clear()
        dataset_status.update({
            'status': 'processing',
            'current': 0,
            'total': len(params['images']) * len(params['models']),
            'model_name': '',
            'results': [],
            'preview_images': []
        })
        
        results = []
        
        # Create output directories with date-based subdirectory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        npz_dir = osp.join(app.config['RESULT_FOLDER'], 'npz', timestamp)
        os.makedirs(npz_dir, exist_ok=True)
        
        if params['generate_vis']:
            vis_dir = osp.join(app.config['RESULT_FOLDER'], 'vis_dataset', timestamp)
            os.makedirs(vis_dir, exist_ok=True)
        
        # Process each model
        for model_idx, model_id in enumerate(params['models']):
            model_name = MODELS[model_id]['name']
            dataset_status['model_name'] = model_name
            
            # Process all images with this model
            predictions = []
            inference_times = []
            
            for img_idx, img_data in enumerate(params['images']):
                img_path, img = img_data
                
                # Update status
                dataset_status['current'] = model_idx * len(params['images']) + img_idx + 1
                
                # Process image
                start_time = time.time()
                depth_map = predict_depth(img, model_id)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Store prediction
                predictions.append(depth_map)
                
                # Generate visualization if requested
                if params['generate_vis'] and img_idx < 10:  # Only first 10 for preview
                    vis_model_dir = osp.join(vis_dir, model_id)
                    os.makedirs(vis_model_dir, exist_ok=True)
                    
                    # Normalize depth for visualization
                    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
                    colored_depth = plt.cm.viridis(depth_norm)[:, :, :3]
                    colored_depth = (colored_depth * 255).astype(np.uint8)
                    
                    # Save visualization
                    vis_path = osp.join(vis_model_dir, f"{img_idx:04d}.png")
                    Image.fromarray(colored_depth).save(vis_path)
            
            # Convert predictions to numpy array
            predictions = np.stack(predictions)
            
            # Save as NPZ file
            npz_filename = f"{model_id}_raw.npz"
            npz_path = osp.join(npz_dir, npz_filename)
            np.savez(npz_path, pred=predictions, pred_type="depth")
            
            # Calculate average inference time
            avg_inference_time = sum(inference_times) / len(inference_times)
            
            # Normalize predictions
            dataset_status['status'] = 'normalizing'
            
            # Apply normalization
            norm_predictions = np.zeros_like(predictions)
            
            if params['normalization'] == 'global':
                # Global min-max normalization
                min_val = np.min(predictions)
                max_val = np.max(predictions)
                norm_predictions = (predictions - min_val) / (max_val - min_val + 1e-8)
            
            elif params['normalization'] == 'per_image':
                # Normalize each image independently
                for i in range(predictions.shape[0]):
                    min_val = np.min(predictions[i])
                    max_val = np.max(predictions[i])
                    norm_predictions[i] = (predictions[i] - min_val) / (max_val - min_val + 1e-8)
            
            elif params['normalization'] == 'affine_invariant':
                # Affine-invariant normalization
                for i in range(predictions.shape[0]):
                    depth = predictions[i]
                    mask_valid = depth > 0
                    depth_valid = depth[mask_valid]
                    
                    if len(depth_valid) > 0:
                        d_min = np.percentile(depth_valid, 5)
                        d_max = np.percentile(depth_valid, 95)
                        
                        norm_depth = (depth - d_min) / (d_max - d_min + 1e-8)
                        norm_depth = np.clip(norm_depth, 0, 1)
                        norm_predictions[i] = norm_depth
                    else:
                        # Fallback if no valid depths
                        min_val = np.min(depth)
                        max_val = np.max(depth)
                        norm_predictions[i] = (depth - min_val) / (max_val - min_val + 1e-8)
            
            # Save normalized predictions
            norm_npz_filename = f"{model_id}_{params['normalization']}.npz"
            norm_npz_path = osp.join(npz_dir, norm_npz_filename)
            np.savez(norm_npz_path, pred=norm_predictions, normalization=params['normalization'])
            
            # Add to results
            results.append({
                'id': model_id,
                'name': model_name,
                'avg_time': f"{avg_inference_time:.3f}",
                'npz_path': f"/static/results/npz/{timestamp}/{norm_npz_filename}",
                'raw_npz_path': f"/static/results/npz/{timestamp}/{npz_filename}"
            })
        
        # Prepare preview images
        preview_images = []
        if params['generate_vis']:
            for i in range(min(5, len(params['images']))):  # Show up to 5 preview images
                preview = {
                    'name': f"Sample {i+1}",
                    'original': f"/static/results/vis_dataset/{timestamp}/sample_{i}.jpg",
                    'depth': f"/static/results/vis_dataset/{timestamp}/{params['models'][0]}/{i:04d}.png"
                }
                
                # Save original image
                img_path, img = params['images'][i]
                img_pil = Image.fromarray(img)
                img_pil.save(osp.join(app.config['RESULT_FOLDER'], f"vis_dataset/{timestamp}/sample_{i}.jpg"))
                
                preview_images.append(preview)
        
        # Mark processing as complete
        dataset_status['status'] = 'complete'
        dataset_status['results'] = results
        dataset_status['preview_images'] = preview_images
        
        # Clean up temp directory
        if dataset_temp_dir and osp.exists(dataset_temp_dir):
            shutil.rmtree(dataset_temp_dir)
            dataset_temp_dir = None
        
    except queue.Empty:
        # No dataset to process
        pass
    except Exception as e:
        # Handle errors
        dataset_status['status'] = 'error'
        dataset_status['message'] = str(e)
        print(f"Error processing dataset: {str(e)}")
        
        # Clean up temp directory
        if dataset_temp_dir and osp.exists(dataset_temp_dir):
            shutil.rmtree(dataset_temp_dir)
            dataset_temp_dir = None


@app.route('/process-dataset/start')
def process_dataset_start():
    """
    Start Server-Sent Events connection for dataset processing
    """
    def event_stream():
        # Send initial status
        yield f"data: {json.dumps({'status': 'init_complete'})}\n\n"
        
        # Report status updates
        last_status = None
        while True:
            current_status = dataset_status.copy()  # Make a copy to avoid modification during serialization
            
            # Debug: print the current status to server logs
            if current_status.get('status') == 'complete':
                print(f"Complete status data: models={len(current_status.get('results', []))}, previews={len(current_status.get('preview_images', []))}")
            
            # Only send if status changed
            if current_status != last_status:
                try:
                    # Convert to JSON and send
                    json_data = json.dumps(current_status)
                    yield f"data: {json_data}\n\n"
                    last_status = current_status
                    
                    # If processing complete, add a delay to ensure client receives it, then break
                    if current_status.get('status') == 'complete' or current_status.get('status') == 'error':
                        time.sleep(1)  # Give client time to process the final update
                        break
                        
                except Exception as e:
                    print(f"Error sending status update: {str(e)}")
                    yield f"data: {json.dumps({'status': 'error', 'message': 'Error sending updates'})}\n\n"
                    break
            
            time.sleep(0.5)
    
    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/process-dataset/upload', methods=['POST'])
def process_dataset_upload():
    """
    Upload dataset for processing
    """
    global dataset_temp_dir
    
    try:
        # Create a temporary directory for the dataset
        dataset_temp_dir = tempfile.mkdtemp()
        
        dataset_type = request.form.get('dataset_type')
        selected_models = request.form.getlist('models[]')
        normalization = request.form.get('normalization', 'affine_invariant')
        generate_vis = request.form.get('generate_vis') == 'true'
        
        # Validate models
        if not selected_models:
            return jsonify({'error': 'No models selected'})
        
        valid_models = [m for m in selected_models if m in MODELS]
        if not valid_models:
            return jsonify({'error': 'No valid models selected'})
        
        images = []
        
        # Handle existing dataset
        if dataset_type == 'existing':
            dataset_path = request.form.get('dataset_path')
            dataset_split = request.form.get('dataset_split', 'all')
            
            if not dataset_path or not os.path.exists(dataset_path):
                return jsonify({'error': 'Invalid dataset path'})
            
            # Check for images directory
            if os.path.exists(os.path.join(dataset_path, 'images')):
                image_dir = os.path.join(dataset_path, 'images')
            else:
                image_dir = dataset_path
            
            # Check for split files
            split_files = []
            if dataset_split != 'all' and os.path.exists(os.path.join(dataset_path, 'splits')):
                split_file_path = os.path.join(dataset_path, 'splits', f'{dataset_split}_files.txt')
                if os.path.exists(split_file_path):
                    with open(split_file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                folder, filename = parts[0], parts[1]
                                split_files.append(os.path.join(folder, 'images', filename))
                            elif len(parts) == 1:
                                split_files.append(parts[0])
            
            # Load images based on split or all
            if split_files:
                # Load only images in the split
                for img_path in split_files:
                    full_path = os.path.join(dataset_path, img_path)
                    if os.path.exists(full_path):
                        try:
                            img = Image.open(full_path).convert('RGB')
                            images.append((img_path, np.array(img)))
                        except Exception as e:
                            print(f"Error loading image {full_path}: {str(e)}")
            else:
                # Check dataset structure
                is_nested = False
                for subdir in glob.glob(os.path.join(dataset_path, '*')):
                    if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, 'images')):
                        is_nested = True
                        break
                
                if is_nested:
                    # For nested structure like SYNS-Patches
                    for scene_dir in glob.glob(os.path.join(dataset_path, '*')):
                        if os.path.isdir(scene_dir) and os.path.exists(os.path.join(scene_dir, 'images')):
                            scene_name = os.path.basename(scene_dir)
                            scene_img_dir = os.path.join(scene_dir, 'images')
                            
                            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                                for img_path in glob.glob(os.path.join(scene_img_dir, ext)):
                                    try:
                                        img = Image.open(img_path).convert('RGB')
                                        # Store with path relative to dataset root: scene/images/file.png
                                        rel_path = os.path.join(scene_name, 'images', os.path.basename(img_path))
                                        images.append((rel_path, np.array(img)))
                                    except Exception as e:
                                        print(f"Error loading image {img_path}: {str(e)}")
                else:
                    # For simple structure with images directly in a folder
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                        for img_path in glob.glob(os.path.join(image_dir, ext)):
                            try:
                                img = Image.open(img_path).convert('RGB')
                                images.append((os.path.basename(img_path), np.array(img)))
                            except Exception as e:
                                print(f"Error loading image {img_path}: {str(e)}")
        
        # Handle uploaded images
        elif dataset_type == 'upload':
            if 'custom_images[]' not in request.files:
                return jsonify({'error': 'No images uploaded'})
            
            custom_images = request.files.getlist('custom_images[]')
            
            # Filter for image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            valid_images = [img for img in custom_images if 
                           osp.splitext(img.filename.lower())[1] in image_extensions]
            
            if not valid_images:
                return jsonify({'error': 'No valid image files found'})
            
            # Load images
            images = []
            for img_file in valid_images:
                try:
                    img = Image.open(img_file).convert('RGB')
                    images.append((img_file.filename, np.array(img)))
                except Exception as e:
                    print(f"Error loading image {img_file.filename}: {str(e)}")
        
        # Check if we have images
        if not images:
            return jsonify({'error': 'No valid images found in the dataset'})
        
        # Queue the dataset for processing
        dataset_queue.put({
            'images': images,
            'models': valid_models,
            'normalization': normalization,
            'generate_vis': generate_vis
        })
        
        # Start the processing thread if not already running
        global dataset_thread
        if dataset_thread is None or not dataset_thread.is_alive():
            dataset_thread = threading.Thread(target=process_dataset_worker)
            dataset_thread.daemon = True
            dataset_thread.start()
        
        return jsonify({'success': True, 'image_count': len(images)})
        
    except Exception as e:
        # Clean up temp directory
        if dataset_temp_dir and osp.exists(dataset_temp_dir):
            shutil.rmtree(dataset_temp_dir)
            dataset_temp_dir = None
        
        return jsonify({'error': str(e)})


@app.route('/process-dataset/process', methods=['POST'])
def process_dataset_process():
    """
    Process the uploaded dataset
    """
    # This just triggers the worker thread to continue
    return jsonify({'success': True})


# Import Response for SSE
from flask import Response


# -------------------------------
# Training Routes
# -------------------------------
import subprocess
import uuid
import threading
from queue import Queue

# Global variables for training
training_status = {}
training_events = {}
training_processes = {}

@app.route('/training/start', methods=['POST'])
def start_training():
    """
    Start training a custom model.
    """
    try:
        # Create a unique ID for this training run
        run_id = str(uuid.uuid4())
        
        # Set initial status
        training_status[run_id] = {
            'status': 'preparing',
            'message': 'Preparing dataset and model...'
        }
        
        # Create a queue for events
        training_events[run_id] = Queue()
        
        # Handle dataset upload if provided
        if 'new_dataset' in request.files:
            dataset_file = request.files['new_dataset']
            if dataset_file.filename:
                # Create dataset directory
                dataset_name = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                dataset_dir = os.path.join('custom_datasets', dataset_name)
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Save and extract dataset
                zip_path = os.path.join(dataset_dir, 'dataset.zip')
                dataset_file.save(zip_path)
                
                # Extract ZIP file
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                
                # Check for required directory structure
                if not (os.path.exists(os.path.join(dataset_dir, 'images')) and 
                        os.path.exists(os.path.join(dataset_dir, 'depths'))):
                    return jsonify({'error': 'Uploaded dataset must contain "images" and "depths" directories'})
                
                dataset_path = dataset_dir
            else:
                return jsonify({'error': 'No dataset file provided'})
        else:
            # Use existing dataset
            dataset_path = request.form.get('dataset_path')
            if not dataset_path:
                return jsonify({'error': 'No dataset specified'})
        
        # Start training in a separate thread
        def run_training():
            try:
                # Prepare command arguments
                cmd = [
                    'python', 'train.py',
                    '--dataset_path', dataset_path,
                    '--batch_size', request.form.get('batch_size', '8'),
                    '--epochs', request.form.get('epochs', '50'),
                    '--lr', request.form.get('learning_rate', '0.001'),
                    '--val_split', request.form.get('val_split', '0.1'),
                    '--base_channels', request.form.get('base_channels', '64'),
                ]
                
                # Add early stopping if enabled
                early_stopping = request.form.get('early_stopping', '0')
                if early_stopping != '0':
                    cmd.extend(['--early_stopping', early_stopping])
                
                # Add TensorBoard if enabled
                if request.form.get('use_tensorboard') == 'true':
                    cmd.append('--use_tensorboard')
                
                # Start the training process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Store process
                training_processes[run_id] = process
                
                # Update status to training
                training_status[run_id] = {
                    'status': 'training',
                    'epoch': 0,
                    'total_epochs': int(request.form.get('epochs', '50')),
                    'train_loss': 0.0,
                    'val_loss': 0.0
                }
                
                # Monitor process output
                for line in process.stdout:
                    # Parse output for progress updates
                    if 'Epoch' in line and 'Train Loss' in line:
                        try:
                            # Extract epoch number
                            epoch_str = line.split('Epoch')[1].split('/')[0].strip()
                            epoch = int(epoch_str)
                            
                            # Extract losses
                            train_loss = float(line.split('Train Loss:')[1].split(',')[0].strip())
                            val_loss = float(line.split('Val Loss:')[1].split(',')[0].strip())
                            
                            # Update status
                            training_status[run_id]['status'] = 'training'
                            training_status[run_id]['epoch'] = epoch
                            training_status[run_id]['train_loss'] = train_loss
                            training_status[run_id]['val_loss'] = val_loss
                            
                            # Check for visualization updates
                            viz_path = os.path.join('custom_models', f"unet_{datetime.now().strftime('%Y%m%d')}*", f"depth_viz_epoch_{epoch}.png")
                            viz_files = glob.glob(viz_path)
                            if viz_files:
                                training_status[run_id]['visualization_path'] = '/' + viz_files[0].replace('\\', '/').lstrip('/')
                            
                            # Put update in queue
                            training_events[run_id].put(training_status[run_id].copy())
                        except Exception as e:
                            print(f"Error parsing training output: {str(e)}")
                    
                    # Check for completion
                    if 'Training complete!' in line:
                        # Extract best validation loss
                        try:
                            best_val_loss = float(line.split('Best validation loss:')[1].split('(')[0].strip())
                            
                            # Update status
                            training_status[run_id]['status'] = 'completed'
                            training_status[run_id]['best_val_loss'] = best_val_loss
                            
                            # Put update in queue
                            training_events[run_id].put(training_status[run_id].copy())
                        except Exception as e:
                            print(f"Error parsing completion output: {str(e)}")
                
                # Process complete
                process.wait()
                
                # Check for errors
                if process.returncode != 0:
                    error_output = process.stderr.read()
                    training_status[run_id] = {
                        'status': 'error',
                        'message': f"Training failed with exit code {process.returncode}: {error_output}"
                    }
                    training_events[run_id].put(training_status[run_id].copy())
            except Exception as e:
                # Handle exceptions
                training_status[run_id] = {
                    'status': 'error',
                    'message': str(e)
                }
                training_events[run_id].put(training_status[run_id].copy())
        
        # Start training thread
        train_thread = threading.Thread(target=run_training)
        train_thread.daemon = True
        train_thread.start()
        
        # Return success response
        return jsonify({
            'status': 'started',
            'run_id': run_id
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/training/monitor/<run_id>')
def monitor_training(run_id):
    """
    Monitor training progress with server-sent events.
    """
    def event_stream():
        # Check if run_id exists
        if run_id not in training_events:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Invalid run ID'})}\n\n"
            return
        
        # Send initial status
        yield f"data: {json.dumps(training_status[run_id])}\n\n"
        
        # Monitor events queue
        while True:
            try:
                # Get events from queue with timeout
                try:
                    status = training_events[run_id].get(timeout=1)
                    yield f"data: {json.dumps(status)}\n\n"
                    
                    # If training is complete or failed, end the stream
                    if status['status'] in ['completed', 'error']:
                        break
                except Exception:
                    # No events in queue, continue
                    pass
                
                # Check if process is still running
                if run_id in training_processes:
                    process = training_processes[run_id]
                    if process.poll() is not None and training_events[run_id].empty():
                        # Process ended but no completion message
                        error_output = process.stderr.read() if process.stderr else "Unknown error"
                        yield f"data: {json.dumps({'status': 'error', 'message': f'Training process ended unexpectedly: {error_output}'})}\n\n"
                        break
            
            except Exception as e:
                # Handle exceptions
                yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
                break
        
        # Clean up resources
        if run_id in training_events:
            del training_events[run_id]
        if run_id in training_processes:
            del training_processes[run_id]
        if run_id in training_status:
            del training_status[run_id]
    
    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/training/models')
def get_trained_models():
    """
    Get list of custom trained models.
    """
    custom_models = []
    custom_models_dir = os.path.join(os.getcwd(), 'custom_models', 'trained')
    
    if os.path.exists(custom_models_dir):
        for model_dir in os.listdir(custom_models_dir):
            model_path = os.path.join(custom_models_dir, model_dir)
            
            # Check if it has required files
            if (os.path.isdir(model_path) and 
                os.path.exists(os.path.join(model_path, 'model.pth')) and
                os.path.exists(os.path.join(model_path, 'info.json'))):
                
                # Load model info
                with open(os.path.join(model_path, 'info.json'), 'r') as f:
                    model_info = json.load(f)
                
                # Add model ID
                model_info['id'] = model_dir
                
                custom_models.append(model_info)
    
    return jsonify({'models': custom_models})


@app.route('/use-model/<model_id>')
def use_custom_model(model_id):
    """
    Use a custom trained model for inference.
    """
    # Check if model exists
    model_path = os.path.join('custom_models', 'trained', model_id)
    if not os.path.exists(model_path):
        return redirect('/')
    
    # Load model info
    with open(os.path.join(model_path, 'info.json'), 'r') as f:
        model_info = json.load(f)
    
    # Add model to MODELS dictionary
    MODELS[f"custom_{model_id}"] = {
        'name': model_info['name'],
        'description': model_info['description'],
        'path': os.path.join(model_path, 'model.pth'),
        'type': model_info['type'],
        'input_size': model_info['input_size']
    }
    
    # Redirect to main page
    return redirect('/')


# -------------------------------
# Results Visualization Routes
# -------------------------------
@app.route('/results/view/<folder_name>')
def view_results(folder_name):
    """
    View NPZ files in the specified folder.
    """
    models_exist, missing_models = check_models()

    if not models_exist:
        # If any model files are missing, show the download instructions page
        return render_template('download.html', missing_models=missing_models)
    
    # Get folders in the npz directory for the navigation
    npz_dir = os.path.join(app.config['RESULT_FOLDER'], 'npz')
    folders = []
    selected_folder = None
    
    if os.path.exists(npz_dir):
        for dir_name in os.listdir(npz_dir):
            folder_path = os.path.join(npz_dir, dir_name)
            
            if os.path.isdir(folder_path):
                # Count files
                file_count = len([f for f in os.listdir(folder_path) if f.endswith('.npz')])
                
                # Get folder size
                folder_size = 0
                for dirpath, dirnames, filenames in os.walk(folder_path):
                    for filename in filenames:
                        file_path = os.path.join(dirpath, filename)
                        folder_size += os.path.getsize(file_path)
                
                # Format size
                if folder_size > 1024 * 1024 * 1024:
                    size_str = f"{folder_size / (1024 * 1024 * 1024):.2f} GB"
                elif folder_size > 1024 * 1024:
                    size_str = f"{folder_size / (1024 * 1024):.2f} MB"
                else:
                    size_str = f"{folder_size / 1024:.2f} KB"
                
                # Get creation time
                created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getctime(folder_path)))
                
                folder_info = {
                    'name': dir_name,
                    'path': folder_path,
                    'file_count': file_count,
                    'size': size_str,
                    'created': created
                }
                
                folders.append(folder_info)
                
                # If this is the selected folder, store its info
                if dir_name == folder_name:
                    selected_folder = folder_info
    
    # Sort folders by creation time (newest first)
    folders.sort(key=lambda x: x['name'], reverse=True)
    
    # If folder doesn't exist, redirect to results page
    if not selected_folder:
        return redirect('/results')
    
    # Get NPZ files in the folder
    npz_files = []
    selected_folder_path = os.path.join(npz_dir, folder_name)
    
    if os.path.exists(selected_folder_path):
        for file_name in os.listdir(selected_folder_path):
            if file_name.endswith('.npz'):
                file_path = os.path.join(selected_folder_path, file_name)
                
                # Get file size
                file_size = os.path.getsize(file_path)
                
                # Format size
                if file_size > 1024 * 1024 * 1024:
                    size_str = f"{file_size / (1024 * 1024 * 1024):.2f} GB"
                elif file_size > 1024 * 1024:
                    size_str = f"{file_size / (1024 * 1024):.2f} MB"
                else:
                    size_str = f"{file_size / 1024:.2f} KB"
                
                # Get model and normalization info from filename
                parts = file_name.split('_')
                model_name = parts[0] if len(parts) > 0 else 'unknown'
                norm_type = parts[1].split('.')[0] if len(parts) > 1 else 'raw'
                
                # Get array dimensions by peeking at the NPZ file
                try:
                    with np.load(file_path) as data:
                        if 'pred' in data:
                            pred_shape = data['pred'].shape
                            image_count = pred_shape[0]
                        else:
                            image_count = 'Unknown'
                except Exception as e:
                    print(f"Error loading NPZ file {file_path}: {e}")
                    image_count = 'Error'
                
                npz_files.append({
                    'name': file_name,
                    'path': file_path,
                    'size': size_str,
                    'model': model_name,
                    'normalization': norm_type,
                    'image_count': image_count
                })
    
    # Sort NPZ files by name
    npz_files.sort(key=lambda x: x['name'])
    
    return render_template('results.html', folders=folders, selected_folder=selected_folder, npz_files=npz_files)


@app.route('/results/storage')
def results_storage():
    """
    Get storage usage for results.
    """
    npz_dir = os.path.join(app.config['RESULT_FOLDER'], 'npz')
    total_size = 0
    
    if os.path.exists(npz_dir):
        for dirpath, dirnames, filenames in os.walk(npz_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
    
    # Format size
    if total_size > 1024 * 1024 * 1024:
        size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
    elif total_size > 1024 * 1024:
        size_str = f"{total_size / (1024 * 1024):.2f} MB"
    else:
        size_str = f"{total_size / 1024:.2f} KB"
    
    return jsonify({'usage': size_str})


@app.route('/results/delete/<folder_name>', methods=['POST'])
def delete_results_folder(folder_name):
    """
    Delete a results folder.
    """
    npz_dir = os.path.join(app.config['RESULT_FOLDER'], 'npz')
    folder_path = os.path.join(npz_dir, folder_name)
    
    # Security check - make sure the folder is in the npz directory
    if not os.path.abspath(folder_path).startswith(os.path.abspath(npz_dir)):
        return jsonify({'success': False, 'error': 'Invalid folder path'})
    
    if not os.path.exists(folder_path):
        return jsonify({'success': False, 'error': 'Folder not found'})
    
    try:
        # Delete the folder and all its contents
        shutil.rmtree(folder_path)
        
        # Also check for matching visualization folder
        vis_dir = os.path.join(app.config['RESULT_FOLDER'], 'vis_dataset', folder_name)
        if os.path.exists(vis_dir):
            shutil.rmtree(vis_dir)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/results/data')
def get_npz_data():
    """
    Get data from an NPZ file for visualization.
    """
    file_path = request.args.get('file')
    
    if not file_path:
        return jsonify({'error': 'No file specified'})
    
    # Security check - make sure the file is in the results directory
    if not os.path.abspath(file_path).startswith(os.path.abspath(app.config['RESULT_FOLDER'])):
        return jsonify({'error': 'Invalid file path'})
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'})
    
    try:
        # Backup solution - create a simple depth map for demo purposes if loading fails
        backup_mode = False
        
        print(f"Loading NPZ file: {file_path}")
        try:
            # Try loading with allow_pickle=True first
            data = np.load(file_path, allow_pickle=True)
        except Exception as load_err:
            print(f"First load attempt failed: {str(load_err)}")
            try:
                # Try again without allow_pickle
                data = np.load(file_path)
            except Exception as second_err:
                print(f"Second load attempt failed: {str(second_err)}")
                # Last resort - create synthetic data
                print("Using backup mode with synthetic depth data")
                backup_mode = True
                
        # Handle backup mode with synthetic data
        if backup_mode:
            # Create a synthetic single depth map (gradient pattern)
            height, width = 480, 640
            y, x = np.mgrid[0:height, 0:width]
            depth_map = 0.5 + 0.5 * np.sin(x/30) * np.cos(y/30)
            
            result = [{
                'data': depth_map.flatten().tolist(),
                'width': width,
                'height': height,
                'min': 0.0,
                'max': 1.0
            }]
            
            return jsonify({
                'data': result,
                'total_images': 1,
                'was_limited': False,
                'is_synthetic': True
            })
            
        # Not in backup mode, process the actual file
        # Debug information about file contents
        print(f"NPZ file keys: {list(data.keys())}")
        
        # Check for 'pred' key (most common) or try other keys
        depth_key = None
        if 'pred' in data:
            depth_key = 'pred'
        elif 'depth' in data:
            depth_key = 'depth'
        elif 'predictions' in data:
            depth_key = 'predictions'
        else:
            # If no standard keys found, use the first array
            for key in data.keys():
                try:
                    if isinstance(data[key], np.ndarray):
                        depth_key = key
                        break
                except:
                    continue
                    
        if depth_key is None:
            print("No valid depth key found, using synthetic data")
            # Create a synthetic gradient pattern
            height, width = 480, 640
            y, x = np.mgrid[0:height, 0:width]
            depth_map = 0.5 + 0.5 * np.sin(x/30) * np.cos(y/30)
            
            result = [{
                'data': depth_map.flatten().tolist(),
                'width': width,
                'height': height,
                'min': 0.0,
                'max': 1.0
            }]
            
            return jsonify({
                'data': result,
                'total_images': 1,
                'was_limited': False,
                'is_synthetic': True
            })
            
        print(f"Using depth key: {depth_key}")
        depth_data = data[depth_key]
        
        # Check if we have a sequence of depth maps
        if isinstance(depth_data, np.ndarray):
            print(f"Depth data shape: {depth_data.shape}, dtype: {depth_data.dtype}")
            
            # Check if depth_data needs reshaping
            if depth_data.ndim == 1 and len(depth_data) > 0 and hasattr(depth_data[0], 'shape'):
                # This handles the case where arrays are stored as objects
                print("Reshaping object array to regular ndarray")
                try:
                    # Convert to a proper ndarray
                    shapes = [img.shape for img in depth_data if hasattr(img, 'shape')]
                    if len(shapes) > 0 and all(s == shapes[0] for s in shapes):
                        reshaped_data = np.zeros((len(shapes),) + shapes[0], dtype=np.float32)
                        for i, img in enumerate(depth_data[:len(shapes)]):
                            if hasattr(img, 'shape') and img.shape == shapes[0]:
                                reshaped_data[i] = img.astype(np.float32)
                        depth_data = reshaped_data
                    else:
                        print("Inconsistent shapes in object array")
                except Exception as reshape_err:
                    print(f"Error reshaping: {str(reshape_err)}")
            
            # Check array dimensions
            if depth_data.ndim < 2:
                print(f"Invalid dimensions: {depth_data.ndim}")
                raise ValueError(f"Depth data has invalid dimensions: {depth_data.ndim}")
            
            # Handle single image case
            if depth_data.ndim == 2:
                print("Single image detected, reshaping")
                depth_data = depth_data.reshape(1, *depth_data.shape)
                
            # For large datasets, limit to the first 20 images to reduce memory usage
            if depth_data.shape[0] > 20:
                print(f"Limiting from {depth_data.shape[0]} to 20 images")
                depth_data = depth_data[:20]
                was_limited = True
            else:
                was_limited = False
                
            # Convert to list of dictionaries for JSON serialization
            result = []
            valid_count = 0
            for i in range(depth_data.shape[0]):
                try:
                    # Get depth map for this image
                    depth_map = depth_data[i].astype(np.float32)  # Convert to float32
                    
                    # Handle NaN or infinite values
                    has_nans = np.isnan(depth_map).any()
                    has_infs = np.isinf(depth_map).any()
                    if has_nans or has_infs:
                        print(f"Image {i} has NaNs: {has_nans}, Infs: {has_infs}")
                        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Get min/max (safely)
                    min_depth = float(np.min(depth_map))
                    max_depth = float(np.max(depth_map))
                    
                    # Normalize depth map to 0-1 range
                    if max_depth > min_depth:
                        normalized = (depth_map - min_depth) / (max_depth - min_depth)
                    else:
                        normalized = np.zeros_like(depth_map)
                    
                    # Flatten to 1D array for JSON
                    result.append({
                        'data': normalized.flatten().tolist(),
                        'width': int(depth_map.shape[1]) if depth_map.ndim > 1 else 1,
                        'height': int(depth_map.shape[0]),
                        'min': float(min_depth),
                        'max': float(max_depth)
                    })
                    valid_count += 1
                    
                    # For debugging first image
                    if i == 0:
                        print(f"First image - Shape: {depth_map.shape}, Min: {min_depth}, Max: {max_depth}")
                        
                except Exception as img_error:
                    print(f"Error processing image {i}: {str(img_error)}")
                    # Skip this image
            
            print(f"Successfully processed {valid_count} images")
            
            # If no valid images were processed
            if len(result) == 0:
                print("No valid images could be processed")
                raise ValueError("No valid depth maps could be processed")
            
            # Get total image count from the original data
            try:
                if hasattr(data[depth_key], 'shape'):
                    total_images = int(data[depth_key].shape[0])
                else:
                    total_images = len(data[depth_key])
            except:
                total_images = len(result)
            
            return jsonify({
                'data': result,
                'total_images': total_images,
                'was_limited': was_limited
            })
        else:
            print(f"Depth data is not a numpy array: {type(depth_data)}")
            raise ValueError(f"Depth data has unexpected type: {type(depth_data)}")
            
    except Exception as e:
        import traceback
        print(f"Error loading NPZ file: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error loading file: {str(e)}'})


# -------------------------------
# Run the Flask app
# -------------------------------
if __name__ == '__main__':
    port = 8085  # Changed to avoid conflicts
    print(f"Starting server on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True, threaded=True)
