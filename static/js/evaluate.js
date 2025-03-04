document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('evaluation-form');
    const loadingSection = document.getElementById('loading');
    const resultsSection = document.getElementById('results');
    const metricsResults = document.getElementById('metrics-results');
    const originalImage = document.getElementById('original-image');
    const groundTruthImage = document.getElementById('ground-truth-image');

    // Better formatting for metric values
    function formatMetric(value, isHigherBetter = false) {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) return value;
        
        // Format to 3 decimal places
        const formatted = numValue.toFixed(3);
        
        // Add color coding based on good/bad values
        if (isHigherBetter) {
            if (numValue > 0.9) return `<span class="good-metric">${formatted}</span>`;
            if (numValue > 0.75) return `<span class="ok-metric">${formatted}</span>`;
            return `<span class="bad-metric">${formatted}</span>`;
        } else {
            if (numValue < 0.1) return `<span class="good-metric">${formatted}</span>`;
            if (numValue < 0.2) return `<span class="ok-metric">${formatted}</span>`;
            return `<span class="bad-metric">${formatted}</span>`;
        }
    }

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading indicator
        loadingSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        
        // Get selected metrics
        const selectedMetrics = Array.from(
            document.querySelectorAll('input[name="metrics"]:checked')
        ).map(cb => cb.value);
        
        if (selectedMetrics.length === 0) {
            alert('Please select at least one metric type');
            loadingSection.classList.add('hidden');
            return;
        }
        
        // Create FormData object to send files
        const formData = new FormData();
        formData.append('image', document.getElementById('image-upload').files[0]);
        formData.append('ground_truth', document.getElementById('ground-truth-upload').files[0]);
        formData.append('eval_mode', document.getElementById('eval-mode').value);
        selectedMetrics.forEach(metric => {
            formData.append('metrics[]', metric);
        });
        
        // Send evaluation request
        fetch('/evaluate-ground-truth', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                loadingSection.classList.add('hidden');
                return;
            }
            
            // Display images
            originalImage.src = data.original;
            groundTruthImage.src = data.ground_truth_viz;
            
            // Clear previous results
            metricsResults.innerHTML = '';
            
            // Add results for each model
            data.results.forEach(result => {
                const row = document.createElement('tr');
                
                // Add model name and depth map
                row.innerHTML = `
                    <td>${result.model_name}</td>
                    <td><img src="${result.depth_map}" alt="${result.model_name} depth map" class="table-thumb"></td>
                    <td>${result.inference_time}s</td>
                `;
                
                // Add metrics
                if (result.metrics) {
                    row.innerHTML += `
                        <td>${formatMetric(result.metrics.AbsRel)}</td>
                        <td>${formatMetric(result.metrics.SqRel)}</td>
                        <td>${formatMetric(result.metrics.RMSE)}</td>
                        <td>${formatMetric(result.metrics.LogRMSE)}</td>
                        <td>${formatMetric(result.metrics.delta1, true)}</td>
                        <td>${formatMetric(result.metrics.delta2, true)}</td>
                        <td>${formatMetric(result.metrics.delta3, true)}</td>
                    `;
                } else if (result.error) {
                    row.innerHTML += `<td colspan="7" class="error">${result.error}</td>`;
                }
                
                metricsResults.appendChild(row);
            });
            
            // Show results
            loadingSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during evaluation. Please try again.');
            loadingSection.classList.add('hidden');
        });
    });
});