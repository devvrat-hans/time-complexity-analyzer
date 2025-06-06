# C++ Time Complexity Analyzer

A comprehensive web-based tool for analyzing the time complexity of C++ code. This analyzer uses static code analysis to detect common algorithmic patterns and provides insights into the computational complexity of your functions.

## Features

- **Web-based Interface**: Modern, responsive Flask web application
- **Multiple Input Methods**: Upload C++ files or paste code snippets directly
- **Comprehensive Analysis**: Detects loops, recursion, standard algorithms, and data structures
- **Visual Results**: Interactive charts and detailed breakdowns
- **Export Options**: Save results in JSON or text format
- **Real-time Analysis**: Fast processing with detailed confidence metrics

## Supported Complexity Types

- **O(1)** - Constant time
- **O(log log n)** - Double logarithmic time (Union-Find operations)
- **O(log n)** - Logarithmic time (binary search, tree operations)
- **O(log² n)** - Log-squared time (some tree algorithms)
- **O(√n)** - Square root time (prime factorization, some number theory)
- **O(n)** - Linear time (single pass through input)
- **O(n log n)** - Linearithmic time (efficient sorting, divide-and-conquer)
- **O(n log² n)** - n-log-squared time (some advanced data structures)
- **O(n√n)** - n-square-root time (some graph algorithms)
- **O(n²)** - Quadratic time (nested loops, some DP)
- **O(n² log n)** - n-squared-log time (some geometric algorithms)
- **O(n³)** - Cubic time (triple nested loops, Floyd-Warshall)
- **O(n⁴)** - Quartic time (four nested loops)
- **O(n^k)** - Polynomial time (k nested loops)
- **O(2^n)** - Exponential time (subset generation, recursive fibonacci)
- **O(3^n)** - Base-3 exponential time
- **O(k^n)** - General exponential time
- **O(n^n)** - n-to-the-n time (some recursive algorithms)
- **O(n!)** - Factorial time (permutation generation)
- **O(2^(2^n))** - Double exponential time (Tower of Hanoi, Ackermann)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd time-complexity-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface (Recommended)

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
   - Upload C++ files (supports .cpp, .cc, .cxx, .c++, .hpp, .h, .hxx, .h++)
   - Paste code snippets directly
   - View interactive analysis results
   - Export results in various formats

### Command Line Interface

For batch processing or automation:

```bash
# Analyze a single file
python complexity_analyzer.py path/to/file.cpp

# Analyze all C++ files in a directory
python complexity_analyzer.py path/to/directory/ -r

# Show detailed analysis reasons
python complexity_analyzer.py path/to/file.cpp -v

# Save results to file
python complexity_analyzer.py path/to/file.cpp -o results.txt

# Filter by confidence threshold
python complexity_analyzer.py path/to/file.cpp --min-confidence 0.8
```

## What It Analyzes

### Loop Patterns
- Simple for/while loops → O(n)
- Nested loops → O(n²), O(n³), etc.
- Divide-and-conquer patterns → O(log n)

### Standard Algorithms
- `std::sort()` → O(n log n)
- `std::binary_search()` → O(log n)
- `std::find()` → O(n)
- And many more...

### Recursion
- Single recursive calls → O(n) or O(log n)
- Multiple recursive calls → O(2^n)
- Divide-and-conquer recursion → O(log n)

### Data Structures
- `std::map`, `std::set` → O(log n) operations
- `std::unordered_map`, `std::unordered_set` → O(1) average operations
- `std::vector` → Various complexities based on usage

## Example Analysis

Input C++ code:
```cpp
int binarySearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

Analysis Result:
- **Function**: `binarySearch`
- **Complexity**: `O(log n)`
- **Confidence**: 95%
- **Reasons**: 
  - Found loop pattern with input halving
  - Divide-and-conquer pattern detected

## Web Interface Features

- **File Upload**: Drag and drop or select multiple C++ files
- **Code Editor**: Syntax-highlighted text area for direct code input
- **Interactive Results**: 
  - Complexity distribution charts
  - Detailed function-by-function analysis
  - Confidence indicators
- **Export Options**: Download results as JSON or formatted text
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Project Structure

```
time-complexity-analyzer/
├── app.py                 # Flask web application
├── complexity_analyzer.py # Core analysis engine
├── gui_analyzer.py       # Legacy tkinter GUI (deprecated)
├── templates/            # HTML templates
│   ├── index.html        # Main interface
│   └── help.html         # Help and documentation
├── static/              # Static web assets
│   ├── css/style.css    # Custom styles
│   └── js/app.js        # Frontend JavaScript
├── test_samples.cpp     # Sample C++ code for testing
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Limitations

- Analysis is based on static code analysis
- Results are estimations; actual performance may vary
- Complex template metaprogramming is not fully supported
- Indirect recursion is not detected
- Maximum file size: 16MB per upload

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For help and documentation, visit `/help` in the web interface or refer to the command line help:
```bash
python complexity_analyzer.py --help
```