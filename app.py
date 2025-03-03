import os
import io
import base64
import numpy as np

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

# Import your models, model dictionary, and the predict_depth function
from models import MODELS, predict_depth

from pathlib import Path

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
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


@app.route('/evaluate', methods=['GET'])
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
# Run the Flask app
# -------------------------------
if __name__ == '__main__':
    port = 8083
    print(f"Starting server on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True, threaded=True)
