// JavaScript for C++ Time Complexity Analyzer

let analysisResults = null;
let complexityChart = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    loadComplexityReference();
});

function initializeEventListeners() {
    // File input change event
    document.getElementById('fileInput').addEventListener('change', handleFileSelection);
    
    // Form submissions
    document.getElementById('uploadForm').addEventListener('submit', handleFormSubmit);
    document.getElementById('codeForm').addEventListener('submit', handleFormSubmit);
    
    // Tab switching
    document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(e) {
            clearResults();
        });
    });
}

function handleFileSelection(event) {
    const files = event.target.files;
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';
    
    if (files.length > 0) {
        const listContainer = document.createElement('div');
        listContainer.className = 'border rounded p-3 bg-light';
        
        const title = document.createElement('h6');
        title.textContent = `Selected Files (${files.length}):`;
        title.className = 'mb-2';
        listContainer.appendChild(title);
        
        Array.from(files).forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            
            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-info';
            
            const icon = document.createElement('i');
            icon.className = 'fas fa-file-code file-icon';
            
            const fileName = document.createElement('span');
            fileName.textContent = file.name;
            
            const fileSize = document.createElement('small');
            fileSize.className = 'text-muted ms-2';
            fileSize.textContent = `(${formatFileSize(file.size)})`;
            
            fileInfo.appendChild(icon);
            fileInfo.appendChild(fileName);
            fileInfo.appendChild(fileSize);
            
            const removeBtn = document.createElement('i');
            removeBtn.className = 'fas fa-times remove-file';
            removeBtn.title = 'Remove file';
            removeBtn.onclick = () => removeFile(index);
            
            fileItem.appendChild(fileInfo);
            fileItem.appendChild(removeBtn);
            listContainer.appendChild(fileItem);
        });
        
        fileList.appendChild(listContainer);
    }
}

function removeFile(index) {
    const fileInput = document.getElementById('fileInput');
    const dt = new DataTransfer();
    
    Array.from(fileInput.files).forEach((file, i) => {
        if (i !== index) {
            dt.items.add(file);
        }
    });
    
    fileInput.files = dt.files;
    handleFileSelection({ target: fileInput });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    
    // Show loading spinner
    showLoading();
    hideError();
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            analysisResults = result;
            displayResults(result);
        } else {
            showError(result.error || 'Analysis failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        hideLoading();
    }
}

function showLoading() {
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

function showError(message) {
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorAlert.style.display = 'block';
    
    // Auto-hide after 10 seconds
    setTimeout(() => {
        errorAlert.style.display = 'none';
    }, 10000);
}

function hideError() {
    document.getElementById('errorAlert').style.display = 'none';
}

function clearResults() {
    document.getElementById('resultsSection').style.display = 'none';
    analysisResults = null;
    if (complexityChart) {
        complexityChart.destroy();
        complexityChart = null;
    }
}

function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const summaryStats = document.getElementById('summaryStats');
    const detailedResults = document.getElementById('detailedResults');
    
    // Display summary statistics
    summaryStats.innerHTML = createSummaryStats(data.summary);
    
    // Create complexity distribution chart
    createComplexityChart(data.summary.complexity_distribution);
    
    // Display detailed results
    detailedResults.innerHTML = createDetailedResults(data.results);
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function createSummaryStats(summary) {
    return `
        <div class="col-md-4">
            <div class="stat-card">
                <div class="stat-number">${summary.total_functions}</div>
                <div class="stat-label">Functions Analyzed</div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="stat-card">
                <div class="stat-number">${summary.files_processed}</div>
                <div class="stat-label">Files Processed</div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="stat-card">
                <div class="stat-number">${Object.keys(summary.complexity_distribution).length}</div>
                <div class="stat-label">Complexity Types Found</div>
            </div>
        </div>
    `;
}

function createComplexityChart(distribution) {
    const ctx = document.getElementById('complexityChart');
    if (!ctx) {
        document.getElementById('complexityChart').innerHTML = '<canvas id="complexityChartCanvas"></canvas>';
    }
    
    const canvas = document.getElementById('complexityChartCanvas') || 
                  (() => {
                      const canvas = document.createElement('canvas');
                      canvas.id = 'complexityChartCanvas';
                      document.getElementById('complexityChart').appendChild(canvas);
                      return canvas;
                  })();
    
    const complexityOrder = ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n²)', 'O(n³)', 'O(2^n)', 'O(n!)', 'O(?)'];
    const colors = ['#198754', '#17a2b8', '#0d6efd', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1', '#e83e8c', '#6c757d'];
    
    const labels = [];
    const data = [];
    const backgroundColor = [];
    
    complexityOrder.forEach((complexity, index) => {
        if (distribution[complexity]) {
            labels.push(complexity);
            data.push(distribution[complexity]);
            backgroundColor.push(colors[index]);
        }
    });
    
    if (complexityChart) {
        complexityChart.destroy();
    }
    
    complexityChart = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColor,
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Complexity Distribution',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    position: 'bottom',
                    labels: { padding: 20 }
                }
            }
        }
    });
}

function createDetailedResults(results) {
    let html = '';
    
    results.forEach(fileResult => {
        html += `
            <div class="mb-4">
                <h4 class="text-primary mb-3">
                    <i class="fas fa-file-code me-2"></i>
                    ${fileResult.filename}
                </h4>
                
                ${fileResult.error ? 
                    `<div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Error: ${fileResult.error}
                    </div>` :
                    fileResult.results.length > 0 ?
                        fileResult.results.map(result => createFunctionResult(result)).join('') :
                        '<div class="alert alert-info">No functions found in this file.</div>'
                }
            </div>
        `;
    });
    
    return html;
}

function createFunctionResult(result) {
    const complexityClass = getComplexityClass(result.complexity);
    const confidencePercent = Math.round(result.confidence * 100);
    
    return `
        <div class="function-result fade-in">
            <div class="function-header">
                <div>
                    <div class="function-name">${result.function_name}</div>
                    <small class="text-muted">Line ${result.line_number}</small>
                </div>
                <span class="badge complexity-badge ${complexityClass}">${result.complexity}</span>
            </div>
            
            <div class="mb-2">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <small class="text-muted">Confidence</small>
                    <small class="text-muted">${confidencePercent}%</small>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                </div>
            </div>
            
            ${result.reasons && result.reasons.length > 0 ? `
                <div>
                    <h6 class="mb-2">Analysis Details:</h6>
                    <ul class="reasons-list">
                        ${result.reasons.map(reason => `<li>${reason}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        </div>
    `;
}

function getComplexityClass(complexity) {
    const mapping = {
        'O(1)': 'bg-success',
        'O(log n)': 'bg-info',
        'O(n)': 'bg-primary',
        'O(n log n)': 'bg-warning text-dark',
        'O(n²)': 'bg-orange',
        'O(n³)': 'bg-danger',
        'O(2^n)': 'bg-danger',
        'O(n!)': 'bg-danger',
        'O(?)': 'bg-secondary'
    };
    return mapping[complexity] || 'bg-secondary';
}

async function exportResults(format) {
    if (!analysisResults) {
        alert('No results to export');
        return;
    }
    
    try {
        if (format === 'json') {
            const blob = new Blob([JSON.stringify(analysisResults, null, 2)], {
                type: 'application/json'
            });
            downloadBlob(blob, 'complexity_analysis.json');
        } else if (format === 'txt') {
            const textContent = formatResultsAsText(analysisResults);
            const blob = new Blob([textContent], { type: 'text/plain' });
            downloadBlob(blob, 'complexity_analysis.txt');
        }
    } catch (error) {
        showError('Export failed: ' + error.message);
    }
}

function formatResultsAsText(data) {
    let text = 'C++ TIME COMPLEXITY ANALYSIS REPORT\n';
    text += '='.repeat(50) + '\n\n';
    
    text += `Analysis Summary:\n`;
    text += `- Total functions analyzed: ${data.summary.total_functions}\n`;
    text += `- Files processed: ${data.summary.files_processed}\n\n`;
    
    text += 'Complexity Distribution:\n';
    Object.entries(data.summary.complexity_distribution).forEach(([complexity, count]) => {
        const percentage = ((count / data.summary.total_functions) * 100).toFixed(1);
        text += `- ${complexity}: ${count} functions (${percentage}%)\n`;
    });
    text += '\n';
    
    text += 'Detailed Results:\n';
    text += '='.repeat(50) + '\n\n';
    
    data.results.forEach(fileResult => {
        text += `File: ${fileResult.filename}\n`;
        text += '-'.repeat(30) + '\n';
        
        if (fileResult.error) {
            text += `Error: ${fileResult.error}\n\n`;
        } else if (fileResult.results.length > 0) {
            fileResult.results.forEach(result => {
                text += `Function: ${result.function_name} (Line ${result.line_number})\n`;
                text += `Complexity: ${result.complexity}\n`;
                text += `Confidence: ${Math.round(result.confidence * 100)}%\n`;
                if (result.reasons && result.reasons.length > 0) {
                    text += `Reasons:\n`;
                    result.reasons.forEach(reason => {
                        text += `  - ${reason}\n`;
                    });
                }
                text += '\n';
            });
        } else {
            text += 'No functions found in this file.\n\n';
        }
    });
    
    return text;
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

async function loadComplexityReference() {
    try {
        const response = await fetch('/api/complexity-types');
        const complexityTypes = await response.json();
        
        // Store for potential modal display
        window.complexityReference = complexityTypes;
    } catch (error) {
        console.warn('Failed to load complexity reference:', error);
    }
}
