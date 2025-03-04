document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('image-upload');
    const previewContainer = document.getElementById('preview-container');
    const evaluateBtn = document.getElementById('evaluate-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');
    const evaluationSection = document.getElementById('evaluation-section');
    
    // Preview selected image
    fileInput.addEventListener('change', function() {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewContainer.innerHTML = `<img src="${e.target.result}" class="img-fluid" alt="Preview">`;
                evaluateBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Process image form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(form);
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        evaluationSection.style.display = 'none';
        resultsSection.style.display = 'none';
        
        // Submit form data for processing
        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Update the results section
            document.getElementById('model-used').textContent = data.model_used;
            document.getElementById('inference-time').textContent = data.inference_time;
            document.getElementById('depth-range').textContent = `${data.depth_min} to ${data.depth_max}`;
            
            // Display the comparison image
            document.getElementById('comparison-container').innerHTML = `
                <img src="${data.comparison}?t=${new Date().getTime()}" class="img-fluid" alt="Depth comparison">
            `;
            
            // Show results section
            resultsSection.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            loadingOverlay.style.display = 'none';
            alert('An error occurred during processing');
        });
    });
    
    // Evaluate all models button
    evaluateBtn.addEventListener('click', function() {
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        evaluationSection.style.display = 'none';
        resultsSection.style.display = 'none';
        
        // Send evaluation request
        fetch('/evaluate-all-models', {
            method: 'GET'
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Clear previous results
            const evaluationResults = document.getElementById('evaluation-results');
            evaluationResults.innerHTML = '';
            
            // Add results for each model
            data.results.forEach(result => {
                const card = document.createElement('div');
                card.className = 'col-md-6 col-lg-4 mb-4';
                card.innerHTML = `
                    <div class="card shadow-sm h-100">
                        <div class="card-header bg-secondary text-white">
                            <h4 class="card-title h6 mb-0">${result.model_name}</h4>
                        </div>
                        <div class="card-body p-0">
                            <img src="${result.depth_map}?t=${new Date().getTime()}" class="img-fluid" alt="${result.model_name} depth map">
                        </div>
                        <div class="card-footer small">
                            <div class="d-flex justify-content-between">
                                <span>Inference: ${result.inference_time}s</span>
                                <span>Range: ${result.depth_min}-${result.depth_max}</span>
                            </div>
                        </div>
                    </div>
                `;
                evaluationResults.appendChild(card);
            });
            
            // Show evaluation section
            evaluationSection.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            loadingOverlay.style.display = 'none';
            alert('An error occurred during evaluation');
        });
    });
});
