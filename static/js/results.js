document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const storageUsage = document.getElementById('storage-usage');
    const confirmDelete = document.getElementById('confirm-delete');
    const deleteModal = document.getElementById('deleteModal');
    const npzSelectors = document.querySelectorAll('.npz-selector');
    const depthViewer = document.getElementById('depthViewer');
    const imageCounter = document.getElementById('image-counter');
    const loadingIndicator = document.getElementById('loading-indicator');
    const colormapSelect = document.getElementById('colormap-select');
    
    // Navigation buttons
    const btnFirst = document.getElementById('btn-first');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    const btnLast = document.getElementById('btn-last');
    
    // Variables
    let currentFolder = null;
    let currentFile = null;
    let depthData = null;
    let currentIndex = 0;
    let totalImages = 0;
    
    // Get storage usage
    fetch('/results/storage')
        .then(response => response.json())
        .then(data => {
            storageUsage.textContent = `Storage: ${data.usage}`;
        })
        .catch(error => console.error('Error fetching storage usage:', error));
    
    // Setup delete folder functionality
    let folderToDelete = null;
    
    document.querySelectorAll('.delete-folder').forEach(button => {
        button.addEventListener('click', function() {
            folderToDelete = this.dataset.folder;
        });
    });
    
    confirmDelete.addEventListener('click', function() {
        if (folderToDelete) {
            fetch(`/results/delete/${folderToDelete}`, {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Reload page after successful deletion
                    window.location.reload();
                } else {
                    alert(`Error: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Error deleting folder:', error);
                alert('An error occurred during deletion.');
            });
        }
        
        // Close modal
        const bsModal = bootstrap.Modal.getInstance(deleteModal);
        bsModal.hide();
    });
    
    // Setup NPZ file selection
    npzSelectors.forEach(selector => {
        selector.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Update active state
            npzSelectors.forEach(item => item.classList.remove('active'));
            this.classList.add('active');
            
            // Get file path
            const filePath = this.dataset.file;
            loadNPZFile(filePath);
        });
    });
    
    // Initialize first NPZ file if available
    if (npzSelectors.length > 0) {
        const firstSelector = npzSelectors[0];
        const filePath = firstSelector.dataset.file;
        loadNPZFile(filePath);
    }
    
    // Load NPZ file data
    function loadNPZFile(filePath) {
        // Show loading indicator
        if (depthViewer) depthViewer.style.display = 'none';
        if (loadingIndicator) loadingIndicator.classList.remove('d-none');
        
        // Reset current data
        currentFile = filePath;
        depthData = null;
        currentIndex = 0;
        
        // Fetch NPZ data
        fetch(`/results/data?file=${encodeURIComponent(filePath)}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    console.error("Server returned error:", data.error);
                    // Show friendly error message to user
                    const errorMessage = document.createElement('div');
                    errorMessage.className = 'alert alert-danger';
                    errorMessage.innerHTML = `
                        <h5>Error Loading File</h5>
                        <p>${data.error}</p>
                        <p>This may be due to an incompatible NPZ file format.</p>
                    `;
                    
                    // Replace loading indicator with error message
                    if (loadingIndicator && loadingIndicator.parentNode) {
                        loadingIndicator.parentNode.replaceChild(errorMessage, loadingIndicator);
                    }
                    return;
                }
                
                // Check if this is synthetic data
                if (data.is_synthetic) {
                    console.log("Using synthetic data pattern");
                    // Optionally show a warning to the user
                }
                
                // Store depth data
                depthData = data.data;
                totalImages = depthData.length;
                
                if (data.was_limited) {
                    console.log(`Note: Dataset was limited to first ${depthData.length} images of ${data.total_images} total`);
                }
                
                // Update counter
                updateImageCounter();
                
                // Render first image
                renderDepthImage(0);
                
                // Hide loading indicator
                if (depthViewer) depthViewer.style.display = 'block';
                if (loadingIndicator) loadingIndicator.classList.add('d-none');
            })
            .catch(error => {
                console.error('Error loading NPZ file:', error);
                
                // Create nice error message
                const errorMessage = document.createElement('div');
                errorMessage.className = 'alert alert-danger';
                errorMessage.innerHTML = `
                    <h5>Error Loading Depth Data</h5>
                    <p>${error.message || 'An unexpected error occurred'}</p>
                    <p>Try using the Dataset Processing feature to create new NPZ files.</p>
                `;
                
                // Replace loading indicator with error message
                if (loadingIndicator && loadingIndicator.parentNode) {
                    loadingIndicator.parentNode.replaceChild(errorMessage, loadingIndicator);
                } else {
                    alert('Failed to load the depth data.');
                }
            });
    }
    
    // Update image counter
    function updateImageCounter() {
        if (imageCounter && depthData) {
            imageCounter.textContent = `Image ${currentIndex + 1} of ${totalImages}`;
            
            // Update button states
            if (btnFirst) btnFirst.disabled = currentIndex === 0;
            if (btnPrev) btnPrev.disabled = currentIndex === 0;
            if (btnNext) btnNext.disabled = currentIndex === totalImages - 1;
            if (btnLast) btnLast.disabled = currentIndex === totalImages - 1;
        }
    }
    
    // Render depth image
    function renderDepthImage(index) {
        if (!depthData || !depthViewer) return;
        
        const ctx = depthViewer.getContext('2d');
        const colormap = colormapSelect ? colormapSelect.value : 'viridis';
        
        // Get depth map data
        const depthMap = depthData[index];
        const width = depthMap.width;
        const height = depthMap.height;
        
        // Resize canvas if needed
        if (depthViewer.width !== width || depthViewer.height !== height) {
            depthViewer.width = width;
            depthViewer.height = height;
        }
        
        // Create image data
        const imageData = ctx.createImageData(width, height);
        
        // Apply colormap based on selection
        for (let i = 0; i < depthMap.data.length; i++) {
            const depth = depthMap.data[i];
            const [r, g, b] = applyColormap(depth, colormap);
            
            // Set RGBA values (alpha always 255)
            const idx = i * 4;
            imageData.data[idx] = r;
            imageData.data[idx + 1] = g;
            imageData.data[idx + 2] = b;
            imageData.data[idx + 3] = 255;
        }
        
        // Put image data on canvas
        ctx.putImageData(imageData, 0, 0);
        
        // Update current index
        currentIndex = index;
        updateImageCounter();
    }
    
    // Apply colormap function
    function applyColormap(value, map = 'viridis') {
        // Clamp value to 0-1
        value = Math.max(0, Math.min(1, value));
        
        // Define colormaps
        const colormaps = {
            viridis: (v) => {
                // Simple approximation of viridis colormap
                return [
                    Math.floor(70 + v * 100),
                    Math.floor(50 + v * 200),
                    Math.floor(120 + v * 135)
                ];
            },
            magma: (v) => {
                return [
                    Math.floor(v < 0.5 ? v * 510 : 255),
                    Math.floor(v < 0.5 ? v * 100 : 50 + v * 150),
                    Math.floor(v < 0.4 ? v * 200 : 80 + v * 175)
                ];
            },
            inferno: (v) => {
                return [
                    Math.floor(v < 0.5 ? v * 500 : 250),
                    Math.floor(v < 0.6 ? v * 50 : 30 + v * 200),
                    Math.floor(v < 0.3 ? 0 : (v - 0.3) * 364)
                ];
            },
            plasma: (v) => {
                return [
                    Math.floor(v < 0.5 ? v * 500 : 250),
                    Math.floor(20 + v * 200),
                    Math.floor(v < 0.5 ? 100 + v * 155 : 100)
                ];
            },
            turbo: (v) => {
                return [
                    Math.floor(v < 0.5 ? v * 400 : 200 - (v - 0.5) * 400),
                    Math.floor(v < 0.3 ? v * 700 : 210 - (v - 0.3) * 210),
                    Math.floor(v < 0.5 ? 0 : (v - 0.5) * 500)
                ];
            },
            jet: (v) => {
                // Simplified jet colormap
                const r = v < 0.4 ? 0 : (v < 0.8 ? (v - 0.4) * 510 : 255);
                const g = v < 0.2 ? v * 510 : (v < 0.6 ? 255 : 255 - (v - 0.6) * 510);
                const b = v < 0.6 ? (v < 0.2 ? 255 : 255 - (v - 0.2) * 510) : 0;
                return [Math.floor(r), Math.floor(g), Math.floor(b)];
            }
        };
        
        return (colormaps[map] || colormaps.viridis)(value);
    }
    
    // Setup navigation buttons
    if (btnFirst) btnFirst.addEventListener('click', () => renderDepthImage(0));
    if (btnPrev) btnPrev.addEventListener('click', () => renderDepthImage(Math.max(0, currentIndex - 1)));
    if (btnNext) btnNext.addEventListener('click', () => renderDepthImage(Math.min(totalImages - 1, currentIndex + 1)));
    if (btnLast) btnLast.addEventListener('click', () => renderDepthImage(totalImages - 1));
    
    // Setup colormap selector
    if (colormapSelect) {
        colormapSelect.addEventListener('change', () => renderDepthImage(currentIndex));
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        if (!depthData) return;
        
        switch(e.key) {
            case 'ArrowLeft':
                renderDepthImage(Math.max(0, currentIndex - 1));
                break;
            case 'ArrowRight':
                renderDepthImage(Math.min(totalImages - 1, currentIndex + 1));
                break;
            case 'Home':
                renderDepthImage(0);
                break;
            case 'End':
                renderDepthImage(totalImages - 1);
                break;
        }
    });
});