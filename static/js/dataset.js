document.addEventListener('DOMContentLoaded', function() {
    const datasetForm = document.getElementById('dataset-form');
    const datasetSelect = document.getElementById('dataset-select');
    const datasetSplit = document.getElementById('dataset-split');
    const customUpload = document.getElementById('custom-upload');
    const processBtn = document.getElementById('process-btn');
    const statusContainer = document.getElementById('status-container');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressLabel = document.getElementById('progress-label');
    const resultsCard = document.getElementById('results-card');
    const resultsContainer = document.getElementById('results-container');
    const previewSection = document.getElementById('preview-section');
    const previewContainer = document.getElementById('preview-container');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    
    // Toggle between dataset selection and upload
    datasetSelect.addEventListener('change', function() {
        if (this.value) {
            customUpload.disabled = true;
            customUpload.value = '';
        } else {
            customUpload.disabled = false;
        }
    });
    
    customUpload.addEventListener('change', function() {
        if (this.files && this.files.length > 0) {
            datasetSelect.disabled = true;
            datasetSelect.value = '';
        } else {
            datasetSelect.disabled = false;
        }
    });
    
    // Process button click handler
    processBtn.addEventListener('click', function() {
        // Validate form
        if (!datasetSelect.value && (!customUpload.files || customUpload.files.length === 0)) {
            showStatus('Error: Please select a dataset or upload images', 'danger');
            return;
        }
        
        // Get selected models
        const selectedModels = Array.from(document.querySelectorAll('.model-checkbox:checked'))
            .map(checkbox => checkbox.value);
        
        if (selectedModels.length === 0) {
            showStatus('Error: Please select at least one model', 'danger');
            return;
        }
        
        // Show loading state
        loadingOverlay.style.display = 'flex';
        showStatus('Initializing dataset processing...', 'info');
        progressContainer.classList.remove('d-none');
        resultsCard.classList.add('d-none');
        previewSection.classList.add('d-none');
        
        // Prepare form data
        const formData = new FormData();
        
        if (datasetSelect.value) {
            formData.append('dataset_path', datasetSelect.value);
            formData.append('dataset_split', datasetSplit.value);
            formData.append('dataset_type', 'existing');
        } else {
            // For custom folder, append all files
            const files = customUpload.files;
            for (let i = 0; i < files.length; i++) {
                formData.append('custom_images[]', files[i]);
            }
            formData.append('dataset_type', 'upload');
        }
        
        // Add other parameters
        selectedModels.forEach(model => {
            formData.append('models[]', model);
        });
        
        formData.append('normalization', document.getElementById('normalization').value);
        formData.append('generate_vis', document.getElementById('generate-vis').checked);
        
        // Start the processing request
        let startTime = new Date().getTime();
        const eventSource = new EventSource('/process-dataset/start');
        
        eventSource.onmessage = function(event) {
            console.log("SSE message received:", event.data.substring(0, 100) + "...");
            try {
                const data = JSON.parse(event.data);
                console.log("Parsed data status:", data.status);
                
                if (data.status === 'init_complete') {
                    // Dataset initialization complete, upload files
                    fetch('/process-dataset/upload', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            showStatus(`Error: ${data.error}`, 'danger');
                            eventSource.close();
                            loadingOverlay.style.display = 'none';
                            return;
                        }
                        
                        showStatus(`Dataset uploaded successfully. Starting processing...`, 'info');
                        fetch('/process-dataset/process', {
                            method: 'POST'
                        });
                    })
                    .catch(error => {
                        showStatus(`Error: ${error.message}`, 'danger');
                        eventSource.close();
                        loadingOverlay.style.display = 'none';
                    });
                } 
                else if (data.status === 'processing') {
                // Update progress
                const percent = Math.round((data.current / data.total) * 100);
                progressBar.style.width = `${percent}%`;
                
                // Calculate ETA
                const elapsedTime = (new Date().getTime() - startTime) / 1000; // in seconds
                const timePerItem = elapsedTime / data.current;
                const remainingItems = data.total - data.current;
                const eta = Math.round(timePerItem * remainingItems);
                
                // Format ETA
                let etaText = '';
                if (eta > 60) {
                    const minutes = Math.floor(eta / 60);
                    const seconds = eta % 60;
                    etaText = `ETA: ${minutes}m ${seconds}s`;
                } else {
                    etaText = `ETA: ${eta}s`;
                }
                
                progressText.textContent = `${percent}% (${data.current}/${data.total}) - ${etaText}`;
                progressLabel.textContent = `Processing model: ${data.model_name}`;
                loadingText.textContent = `Processing ${data.current}/${data.total} images with ${data.model_name}...`;
            }
            else if (data.status === 'normalizing') {
                progressLabel.textContent = `Normalizing results: ${data.model_name}`;
                loadingText.textContent = `Normalizing results for ${data.model_name}...`;
            }
            else if (data.status === 'complete') {
                // Processing complete
                eventSource.close();
                loadingOverlay.style.display = 'none';
                
                // Calculate elapsed time
                const elapsed = Math.round((new Date().getTime() - startTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                const timeString = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
                
                // Show results
                showStatus(`Processing complete! Elapsed time: ${timeString}`, 'success');
                
                console.log("Complete data:", data);
                console.log("Results models:", data.results ? data.results.length : 0);
                console.log("Preview images:", data.preview_images ? data.preview_images.length : 0);
                
                // Display results
                resultsCard.classList.remove('d-none');
                resultsContainer.innerHTML = '';
                
                // Add download links for each model
                let resultsHtml = '<h4 class="h6 mb-3">Download Prediction Files:</h4><div class="list-group">';
                
                if (data.results && data.results.length > 0) {
                    data.results.forEach(model => {
                        resultsHtml += `
                            <div class="list-group-item d-flex justify-content-between align-items-center">
                                <div>
                                    <strong>${model.name}</strong>
                                    <small class="d-block text-muted">Avg. inference: ${model.avg_time}s</small>
                                </div>
                                <div>
                                    <a href="${model.npz_path}" class="btn btn-sm btn-primary" download>Download NPZ</a>
                                </div>
                            </div>
                        `;
                    });
                } else {
                    resultsHtml += '<div class="alert alert-warning">No model results available.</div>';
                }
                
                resultsHtml += '</div>';
                resultsContainer.innerHTML = resultsHtml;
                
                // Show preview images if available
                if (data.preview_images && data.preview_images.length > 0) {
                    previewSection.classList.remove('d-none');
                    previewContainer.innerHTML = '';
                    
                    // Display a grid of preview images
                    data.preview_images.forEach(preview => {
                        const previewCard = document.createElement('div');
                        previewCard.className = 'col-md-4 mb-4';
                        previewCard.innerHTML = `
                            <div class="card h-100">
                                <div class="card-header">
                                    <h5 class="card-title h6">${preview.name}</h5>
                                </div>
                                <div class="card-body p-0">
                                    <div class="row g-0">
                                        <div class="col-6">
                                            <img src="${preview.original}" class="img-fluid" alt="Original">
                                            <div class="text-center small py-1">Original</div>
                                        </div>
                                        <div class="col-6">
                                            <img src="${preview.depth}" class="img-fluid" alt="Depth">
                                            <div class="text-center small py-1">Depth</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                        previewContainer.appendChild(previewCard);
                    });
                } else {
                    previewSection.classList.add('d-none');
                }
            }
            else if (data.status === 'error') {
                // Error during processing
                eventSource.close();
                loadingOverlay.style.display = 'none';
                showStatus(`Error: ${data.message}`, 'danger');
            }
            } catch (e) {
                console.error("Error parsing SSE data:", e);
                showStatus(`Error parsing server response: ${e.message}`, 'danger');
                eventSource.close();
                loadingOverlay.style.display = 'none';
            }
        };
        
        eventSource.onerror = function() {
            eventSource.close();
            loadingOverlay.style.display = 'none';
            showStatus('Error: Connection to server lost', 'danger');
        };
    });
    
    // Helper function to show status messages
    function showStatus(message, type = 'info') {
        const statusClass = type === 'info' ? 'text-info' : 
                           type === 'success' ? 'text-success' : 
                           type === 'warning' ? 'text-warning' : 'text-danger';
        
        statusContainer.innerHTML = `<p class="${statusClass}">${message}</p>`;
    }
});