# Monocular Depth Estimation Demo

A web application that demonstrates monocular depth estimation using various state-of-the-art models.

## Features

- Upload and process images to generate accurate depth maps
- Multiple depth estimation models with different characteristics
- Side-by-side comparison of original image and depth map
- Model evaluation mode to compare results from all models
- Performance metrics (inference time, depth range)
- Responsive web interface

## Models

The application supports the following depth estimation models:

1. **MiDaS Small** - Lightweight model for fast inference (ONNX format)
2. **DPT Hybrid** - Good balance between speed and accuracy
3. **DPT Large** - Highest quality depth estimation
4. **MonoDepth2** - Accurate depth with self-supervised training

## Installation

1. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download the pre-trained models:
```
python model_downloader.py
```

## Usage

1. Start the Flask application:
```
python app.py
```

2. Open a web browser and navigate to:
```
http://127.0.0.1:8081/
```

3. Upload an image, select a model, and click "Process Image" to generate a depth map

4. Use the "Evaluate All Models" button to compare results from all available models

## How It Works

1. The user uploads an image through the web interface
2. The selected model processes the image to create a depth map
   - MiDaS Small uses ONNX Runtime for inference
   - DPT models use PyTorch with Transformers
   - MonoDepth2 uses PyTorch with a custom encoder-decoder architecture
3. The depth map is colorized using the viridis colormap
4. Results are displayed alongside the original image
5. Performance metrics (inference time, depth range) are calculated and displayed

## Model Details

- **MiDaS Small**: A compressed version of the MiDaS depth estimation network, optimized for speed.
- **DPT Hybrid**: Dense Prediction Transformer with a hybrid backbone, balancing performance and accuracy.
- **DPT Large**: Full-size Dense Prediction Transformer for highest quality depth maps.
- **MonoDepth2**: Self-supervised monocular depth estimation model with good generalization capabilities.

## Requirements

- Python 3.6+
- Flask
- PyTorch & TorchVision
- Transformers & Hugging Face Hub
- ONNX & ONNX Runtime
- OpenCV
- NumPy & Matplotlib
- Pillow
- timm (for vision models)
- tqdm (for download progress)