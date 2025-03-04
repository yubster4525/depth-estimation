import os
import io
import base64
import numpy as np
import json
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

    return render_template('index.html', models=template_models)


@app.route('/dataset')
def dataset_page():
    """
    Dataset evaluation page. Allows batch processing of datasets.
    """
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

    return render_template('dataset.html', models=template_models)


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


def evaluate_depth_prediction(pred_depth, gt_depth, eval_mode='mono'):
    """
    Evaluate a depth prediction against ground truth.
    
    Args:
        pred_depth (numpy.ndarray): Predicted depth map.
        gt_depth (numpy.ndarray): Ground truth depth map.
        eval_mode (str): Evaluation mode ('mono' or 'stereo').
            - 'mono': Scale-invariant evaluation (align pred to gt).
            - 'stereo': Fixed-scale evaluation.
            
    Returns:
        dict: Dictionary of metrics.
    """
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
    
    # Compute metrics
    metrics = metrics_eigen(pred_depth, gt_depth, mask)
    
    return metrics


@app.route('/evaluate', methods=['GET'])
def evaluate_page():
    """
    Render the ground truth evaluation page.
    """
    return render_template('evaluate.html')


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
        
        # Process with all models
        results = []
        
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
                
                # Compute metrics
                metrics = evaluate_depth_prediction(pred_depth, gt_depth, eval_mode)
                
                print(f"Model {model_id} metrics: {metrics}")
                
                results.append({
                    'model_id': model_id,
                    'model_name': MODELS[model_id]['name'],
                    'inference_time': f"{inference_time:.2f}",
                    'depth_map': os.path.join('static', 'results', f'depth_{model_id}.jpg'),
                    'depth_min': f"{depth_min:.2f}",
                    'depth_max': f"{depth_max:.2f}",
                    'metrics': metrics
                })
            except Exception as e:
                print(f"Error evaluating model {model_id}: {str(e)}")
                results.append({
                    'model_id': model_id,
                    'model_name': MODELS[model_id]['name'],
                    'error': str(e)
                })
        
        # Return results
        return jsonify({
            'success': True,
            'original': os.path.join('static', 'uploads', 'eval_input.jpg'),
            'ground_truth_viz': os.path.join('static', 'ground_truth', 'ground_truth_viz.jpg'),
            'results': results
        })
        
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
        
        dataset_status['status'] = 'processing'
        dataset_status['current'] = 0
        dataset_status['total'] = len(params['images']) * len(params['models'])
        
        results = []
        
        # Create output directories
        npz_dir = osp.join(app.config['RESULT_FOLDER'], 'npz')
        os.makedirs(npz_dir, exist_ok=True)
        
        if params['generate_vis']:
            vis_dir = osp.join(app.config['RESULT_FOLDER'], 'vis_dataset')
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
                'npz_path': f"/static/results/npz/{norm_npz_filename}",
                'raw_npz_path': f"/static/results/npz/{npz_filename}"
            })
        
        # Prepare preview images
        preview_images = []
        if params['generate_vis']:
            for i in range(min(5, len(params['images']))):  # Show up to 5 preview images
                preview = {
                    'name': f"Sample {i+1}",
                    'original': f"/static/results/vis_dataset/sample_{i}.jpg",
                    'depth': f"/static/results/vis_dataset/{params['models'][0]}/{i:04d}.png"
                }
                
                # Save original image
                img_path, img = params['images'][i]
                img_pil = Image.fromarray(img)
                img_pil.save(osp.join(app.config['RESULT_FOLDER'], f"vis_dataset/sample_{i}.jpg"))
                
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
            if dataset_status != last_status:
                yield f"data: {json.dumps(dataset_status)}\n\n"
                last_status = dataset_status.copy()
            
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
        
        # Handle SYNS dataset
        if dataset_type == 'syns':
            if 'syns_zip' not in request.files:
                return jsonify({'error': 'No SYNS zip file uploaded'})
            
            syns_zip = request.files['syns_zip']
            syns_split = request.form.get('syns_split', 'val')
            
            # Save the zip file
            zip_path = osp.join(dataset_temp_dir, 'syns_patches.zip')
            syns_zip.save(zip_path)
            
            # Initialize SYNS accessor
            from io import BytesIO
            import zipfile
            
            # Load images from the zip
            images = []
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                # Load split files
                split_files_path = f"syns_patches/splits/{syns_split}_files.txt"
                if split_files_path not in zip_file.namelist():
                    return jsonify({'error': f"Split file '{split_files_path}' not found in zip"})
                
                split_files_content = zip_file.read(split_files_path).decode("utf-8")
                split_files = split_files_content.splitlines()
                
                # Load images
                for line in split_files:
                    folder_name, image_name = line.split()
                    image_path = f"syns_patches/{folder_name}/images/{image_name}"
                    
                    if image_path not in zip_file.namelist():
                        return jsonify({'error': f"Image '{image_path}' not found in zip"})
                    
                    image_data = zip_file.read(image_path)
                    image = Image.open(BytesIO(image_data)).convert('RGB')
                    images.append((image_path, np.array(image)))
        
        # Handle custom image folder
        else:
            if 'custom_images[]' not in request.files:
                return jsonify({'error': 'No custom images uploaded'})
            
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


# -------------------------------
# Run the Flask app
# -------------------------------
if __name__ == '__main__':
    port = 8085  # Changed to avoid conflicts
    print(f"Starting server on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True, threaded=True)
