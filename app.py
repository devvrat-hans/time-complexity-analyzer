#!/usr/bin/env python3
"""
Flask Web Interface for C++ Time Complexity Analyzer
A modern web-based frontend for analyzing C++ code complexity.
"""

import os
import json
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from complexity_analyzer import CppTimeComplexityAnalyzer, ComplexityResult, ComplexityType
import zipfile
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize the analyzer
analyzer = CppTimeComplexityAnalyzer()

ALLOWED_EXTENSIONS = {'.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h', '.hxx', '.h++'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def complexity_to_dict(result: ComplexityResult):
    """Convert ComplexityResult to dictionary for JSON serialization."""
    return {
        'function_name': result.function_name,
        'complexity': result.complexity.value,
        'confidence': result.confidence,
        'reasons': result.reasons,
        'line_number': result.line_number
    }

@app.route('/')
def index():
    """Main page with upload form and analyzer interface."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_code():
    """Analyze uploaded C++ files or code snippet."""
    try:
        analysis_results = []
        
        # Check if it's a code snippet or file upload
        if 'code_snippet' in request.form and request.form['code_snippet'].strip():
            # Analyze code snippet
            code_snippet = request.form['code_snippet']
            filename = request.form.get('snippet_name', 'code_snippet.cpp')
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as tmp_file:
                tmp_file.write(code_snippet)
                tmp_file_path = tmp_file.name
            
            try:
                results = analyzer.analyze_file(tmp_file_path)
                analysis_results.append({
                    'filename': filename,
                    'file_path': filename,
                    'results': [complexity_to_dict(r) for r in results],
                    'error': None
                })
            except Exception as e:
                analysis_results.append({
                    'filename': filename,
                    'file_path': filename,
                    'results': [],
                    'error': str(e)
                })
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        # Check for file uploads
        if 'files' in request.files:
            files = request.files.getlist('files')
            
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    
                    # Save file temporarily
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    try:
                        results = analyzer.analyze_file(file_path)
                        analysis_results.append({
                            'filename': filename,
                            'file_path': filename,
                            'results': [complexity_to_dict(r) for r in results],
                            'error': None
                        })
                    except Exception as e:
                        analysis_results.append({
                            'filename': filename,
                            'file_path': filename,
                            'results': [],
                            'error': str(e)
                        })
                    finally:
                        # Clean up uploaded file
                        if os.path.exists(file_path):
                            os.unlink(file_path)
                
                elif file and file.filename:
                    analysis_results.append({
                        'filename': file.filename,
                        'file_path': file.filename,
                        'results': [],
                        'error': f"File type not supported. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
                    })
        
        if not analysis_results:
            return jsonify({
                'success': False,
                'error': 'No valid files or code snippet provided'
            })
        
        # Calculate summary statistics
        total_functions = sum(len(result['results']) for result in analysis_results)
        complexity_counts = {}
        
        for file_result in analysis_results:
            for result in file_result['results']:
                complexity = result['complexity']
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        return jsonify({
            'success': True,
            'results': analysis_results,
            'summary': {
                'total_functions': total_functions,
                'files_processed': len(analysis_results),
                'complexity_distribution': complexity_counts
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        })

@app.route('/export/<format>')
def export_results(format):
    """Export analysis results in various formats."""
    # This would be called via AJAX with results data
    # For now, return a sample export
    if format == 'json':
        return jsonify({'message': 'Export functionality - implement based on stored results'})
    elif format == 'txt':
        return "Export functionality - implement based on stored results", 200, {
            'Content-Type': 'text/plain',
            'Content-Disposition': 'attachment; filename=analysis_results.txt'
        }
    else:
        return jsonify({'error': 'Unsupported export format'}), 400

@app.route('/help')
def help():
    """Help page with usage instructions."""
    return render_template('help.html')

@app.route('/api/complexity-types')
def get_complexity_types():
    """Get list of all complexity types with descriptions."""
    complexity_info = {
        'O(1)': {
            'name': 'Constant',
            'description': 'Operations that take the same amount of time regardless of input size',
            'examples': ['Array access', 'Hash table lookup', 'Simple calculations']
        },
        'O(log n)': {
            'name': 'Logarithmic',
            'description': 'Time grows logarithmically with input size',
            'examples': ['Binary search', 'Tree operations', 'Divide and conquer algorithms']
        },
        'O(n)': {
            'name': 'Linear',
            'description': 'Time grows linearly with input size',
            'examples': ['Linear search', 'Single loop through array', 'Reading input']
        },
        'O(n log n)': {
            'name': 'Linearithmic',
            'description': 'Efficient sorting algorithms complexity',
            'examples': ['Merge sort', 'Heap sort', 'Quick sort (average case)']
        },
        'O(n²)': {
            'name': 'Quadratic',
            'description': 'Time grows with the square of input size',
            'examples': ['Bubble sort', 'Nested loops', 'Simple matrix operations']
        },
        'O(n³)': {
            'name': 'Cubic',
            'description': 'Time grows with the cube of input size',
            'examples': ['Matrix multiplication', 'Triple nested loops']
        },
        'O(2^n)': {
            'name': 'Exponential',
            'description': 'Time doubles with each additional input element',
            'examples': ['Recursive Fibonacci', 'Subset generation', 'Brute force solutions']
        },
        'O(n!)': {
            'name': 'Factorial',
            'description': 'Time grows factorially with input size',
            'examples': ['Permutation generation', 'Traveling salesman (brute force)']
        }
    }
    return jsonify(complexity_info)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Create static directory if it doesn't exist
    static_dir = Path(__file__).parent / 'static'
    static_dir.mkdir(exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=8081)
