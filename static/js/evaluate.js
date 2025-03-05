document.addEventListener('DOMContentLoaded', function() {
    console.log('Evaluate.js loaded successfully');
    // Check if we're on the evaluate page
    const evaluationForm = document.getElementById('evaluation-form');
    if (!evaluationForm) {
        console.error('Evaluation form not found in the DOM');
    } else {
        console.log('Evaluation form found in DOM');
    }
    const form = document.getElementById('evaluation-form');
    const evaluateBtn = document.getElementById('evaluate-btn');
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
    
    // Add click event listener to the evaluate button
    evaluateBtn.addEventListener('click', function() {
        console.log('Evaluate button clicked, triggering form submit');
        form.dispatchEvent(new Event('submit'));
    });

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading indicator
        loadingSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        
        // Get selected metrics
        const selectedMetrics = Array.from(
            document.querySelectorAll('input[name="metrics"]:checked')
        ).map(cb => cb.value);
        
        // Force at least one metric even if none selected
        if (selectedMetrics.length === 0) {
            selectedMetrics.push('eigen');
        }
        
        // Create FormData object to send files
        const formData = new FormData();
        formData.append('image', document.getElementById('image-upload').files[0]);
        formData.append('ground_truth', document.getElementById('ground-truth-upload').files[0]);
        formData.append('eval_mode', document.getElementById('eval-mode').value);
        selectedMetrics.forEach(metric => {
            formData.append('metrics', metric);
        });
        
        console.log('Submitting with metrics:', selectedMetrics);
        
        // Send evaluation request
        fetch('/evaluate-ground-truth', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Response data:', data);
            
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
                    // Add standard metrics
                    row.innerHTML += `
                        <td>${formatMetric(result.metrics.AbsRel)}</td>
                        <td>${formatMetric(result.metrics.SqRel)}</td>
                        <td>${formatMetric(result.metrics.RMSE)}</td>
                        <td>${formatMetric(result.metrics.LogRMSE)}</td>
                        <td>${formatMetric(result.metrics.delta1, true)}</td>
                        <td>${formatMetric(result.metrics.delta2, true)}</td>
                        <td>${formatMetric(result.metrics.delta3, true)}</td>
                    `;
                    
                    // Check if we have edge metrics
                    if (result.metrics['F-Score'] !== undefined) {
                        // Add F-Score to existing table or dynamically add column if it doesn't exist
                        const headerRow = document.querySelector('#metrics-table thead tr');
                        if (!headerRow.querySelector('th[data-metric="f-score"]')) {
                            // Add F-Score column to header
                            const fScoreHeader = document.createElement('th');
                            fScoreHeader.setAttribute('data-metric', 'f-score');
                            fScoreHeader.innerHTML = 'F-Score ↑';
                            headerRow.appendChild(fScoreHeader);
                            
                            // Add edge visualization column if we have edge maps
                            if (result.edge_map) {
                                const edgeImgHeader = document.createElement('th');
                                edgeImgHeader.innerHTML = 'Edges';
                                headerRow.insertBefore(edgeImgHeader, headerRow.querySelector('th:nth-child(3)'));
                            }
                        }
                        
                        // Add edge visualization to row if it exists
                        if (result.edge_map && !row.querySelector('td:nth-child(3) img')) {
                            const depthCell = row.querySelector('td:nth-child(2)');
                            const edgeCell = document.createElement('td');
                            edgeCell.innerHTML = `<img src="${result.edge_map}" alt="${result.model_name} edges" class="table-thumb">`;
                            row.insertBefore(edgeCell, depthCell.nextSibling);
                        }
                        
                        // Add F-Score cell
                        const fScoreCell = document.createElement('td');
                        fScoreCell.innerHTML = formatMetric(result.metrics['F-Score'], true);
                        row.appendChild(fScoreCell);
                    }
                } else if (result.error) {
                    const colSpan = document.querySelectorAll('#metrics-table thead th').length - 3;
                    row.innerHTML += `<td colspan="${colSpan}" class="error">${result.error}</td>`;
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