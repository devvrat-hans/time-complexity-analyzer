#!/usr/bin/env python3
"""
Time Complexity Analyzer for C++ Code
Analyzes C++ source files to determine their time complexity.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ComplexityType(Enum):
    # Constant and sub-linear complexities
    CONSTANT = "O(1)"
    INVERSE_ACKERMANN = "O(α(n))"  # Inverse Ackermann function
    LOG_STAR = "O(log* n)"  # Iterated logarithm
    LOG_LOG = "O(log log n)"
    LOGARITHMIC = "O(log n)"
    LOG_CUBED = "O(log³ n)"
    LOG_SQUARED = "O(log² n)"
    LOG_N_OVER_LOG_LOG_N = "O(log n / log log n)"
    
    # Sub-linear to linear
    SQRT_LOG = "O(√n log n)"
    SQRT = "O(√n)"
    N_TO_2_3 = "O(n^(2/3))"
    N_TO_3_4 = "O(n^(3/4))"
    N_TO_4_5 = "O(n^(4/5))"
    N_TO_5_6 = "O(n^(5/6))"
    
    # Linear and near-linear
    LINEAR = "O(n)"
    N_LOG_LOG = "O(n log log n)"
    LINEARITHMIC = "O(n log n)"
    N_LOG_SQUARED = "O(n log² n)"
    N_LOG_CUBED = "O(n log³ n)"
    N_SQRT = "O(n√n)"
    N_TO_1_5 = "O(n^1.5)"
    
    # Quadratic and related
    N_SQRT_LOG = "O(n√n log n)"
    N_TO_5_3 = "O(n^(5/3))"
    N_TO_7_4 = "O(n^(7/4))"
    QUADRATIC = "O(n²)"
    N_SQUARED_LOG_LOG = "O(n² log log n)"
    N_SQUARED_LOG = "O(n² log n)"
    N_SQUARED_LOG_SQUARED = "O(n² log² n)"
    N_TO_2_5 = "O(n^2.5)"
    N_TO_8_3 = "O(n^(8/3))"
    
    # Cubic and higher polynomials
    CUBIC = "O(n³)"
    N_CUBED_LOG = "O(n³ log n)"
    N_TO_3_5 = "O(n^3.5)"
    QUARTIC = "O(n⁴)"
    N_TO_4_5_ALT = "O(n^4.5)"
    QUINTIC = "O(n⁵)"
    SEXTIC = "O(n⁶)"
    POLYNOMIAL = "O(n^k)"
    
    # Matrix multiplication complexities
    MATRIX_MULT_NAIVE = "O(n³)"
    MATRIX_MULT_STRASSEN = "O(n^2.807)"
    MATRIX_MULT_COPPERSMITH = "O(n^2.376)"
    MATRIX_MULT_OPTIMAL = "O(n^2.373)"
    
    # Graph algorithm specific complexities
    V_PLUS_E = "O(V + E)"  # DFS, BFS
    E_LOG_V = "O(E log V)"  # Kruskal's, Prim's
    V_LOG_V_PLUS_E = "O((V + E) log V)"  # Dijkstra with binary heap
    V_LOG_V_PLUS_E_ALPHA = "O((V + E) α(V))"  # Dijkstra with Fibonacci heap
    V_E = "O(VE)"  # Bellman-Ford
    V_CUBED = "O(V³)"  # Floyd-Warshall
    E_SQRT_V = "O(E√V)"  # Hopcroft-Karp
    V_SQUARED = "O(V²)"  # Dense graph operations
    
    # String algorithm complexities
    STRING_LENGTH = "O(m)"  # Pattern length
    TEXT_PLUS_PATTERN = "O(n + m)"  # KMP, Z-algorithm
    TEXT_TIMES_PATTERN = "O(nm)"  # Naive string matching
    SUFFIX_ARRAY_CONSTRUCTION = "O(n log n)"
    SUFFIX_TREE_CONSTRUCTION = "O(n)"
    LCS_DP = "O(nm)"  # Longest Common Subsequence
    EDIT_DISTANCE = "O(nm)"  # Levenshtein distance
    
    # Sorting complexities
    COMPARISON_SORT_OPTIMAL = "O(n log n)"
    COUNTING_SORT = "O(n + k)"  # k is range of input
    RADIX_SORT = "O(d(n + k))"  # d is number of digits
    BUCKET_SORT = "O(n + k)"
    
    # Number theory complexities
    SIEVE_OF_ERATOSTHENES = "O(n log log n)"
    PRIME_FACTORIZATION = "O(√n)"
    GCD_EUCLIDEAN = "O(log(min(a,b)))"
    MODULAR_EXPONENTIATION = "O(log b)"  # a^b mod m
    
    # Geometric algorithm complexities
    CONVEX_HULL_2D = "O(n log n)"
    CLOSEST_PAIR = "O(n log n)"
    LINE_INTERSECTION = "O(n log n)"
    VORONOI_DIAGRAM = "O(n log n)"
    DELAUNAY_TRIANGULATION = "O(n log n)"
    
    # Data structure operation complexities
    HASH_TABLE_WORST = "O(n)"
    BST_WORST = "O(n)"
    AVL_OPERATIONS = "O(log n)"
    HEAP_OPERATIONS = "O(log n)"
    TRIE_OPERATIONS = "O(m)"  # m is key length
    SEGMENT_TREE_OPERATIONS = "O(log n)"
    FENWICK_TREE_OPERATIONS = "O(log n)"
    
    # Exponential and super-exponential
    EXPONENTIAL_BASE_2 = "O(2^n)"
    EXPONENTIAL_BASE_3 = "O(3^n)"
    EXPONENTIAL_BASE_E = "O(e^n)"
    EXPONENTIAL_BASE_K = "O(k^n)"
    EXPONENTIAL_LINEAR = "O(n·2^n)"
    EXPONENTIAL_QUADRATIC = "O(n²·2^n)"
    EXPONENTIAL_POLY = "O(n^n)"
    
    # Factorial and super-exponential
    FACTORIAL = "O(n!)"
    FACTORIAL_APPROX = "O(√n·(n/e)^n)"  # Stirling's approximation
    DOUBLE_EXPONENTIAL = "O(2^(2^n))"
    TOWER_FUNCTION = "O(2^2^...^2)"  # Tower of exponentials
    ACKERMANN = "O(A(m,n))"  # Ackermann function
    
    # Special cases and unknowns
    AMORTIZED_CONSTANT = "O(1) amortized"
    EXPECTED_LINEAR = "O(n) expected"
    PROBABILISTIC = "O(f(n)) with high probability"
    UNKNOWN = "O(?)"

@dataclass
class ComplexityResult:
    function_name: str
    complexity: ComplexityType
    confidence: float
    reasons: List[str]
    line_number: int

class CppTimeComplexityAnalyzer:
    def __init__(self):
        self.loop_patterns = {
            # Linear patterns
            r'for\s*\(\s*[^;]*;\s*[^;]*<\s*n\s*;\s*[^)]*\+\+[^)]*\)': ComplexityType.LINEAR,
            r'for\s*\(\s*[^;]*;\s*[^;]*<\s*\w+\.size\(\)\s*;\s*[^)]*\+\+[^)]*\)': ComplexityType.LINEAR,
            r'while\s*\(\s*[^)]*<\s*n\s*\)': ComplexityType.LINEAR,
            
            # Logarithmic patterns
            r'while\s*\(\s*n\s*>\s*0\s*\).*n\s*/=\s*2': ComplexityType.LOGARITHMIC,
            r'while\s*\(\s*n\s*>\s*1\s*\).*n\s*/=\s*2': ComplexityType.LOGARITHMIC,
            r'while\s*\(\s*\w+\s*>\s*0\s*\).*\w+\s*>>=\s*1': ComplexityType.LOGARITHMIC,
            r'for\s*\([^;]*;\s*\w+\s*>\s*0\s*;\s*\w+\s*/=\s*2\s*\)': ComplexityType.LOGARITHMIC,
            
            # Square root patterns
            r'for\s*\([^;]*;\s*\w+\s*\*\s*\w+\s*<=\s*n\s*;\s*\w+\+\+\s*\)': ComplexityType.SQRT,
            r'while\s*\(\s*\w+\s*\*\s*\w+\s*<=\s*n\s*\)': ComplexityType.SQRT,
            
            # Log squared patterns (often in segment trees, sparse tables)
            r'for\s*\([^;]*;\s*\w+\s*<\s*log2?\s*\(\s*n\s*\)\s*;\s*\w+\+\+\s*\).*for': ComplexityType.LOG_SQUARED,
        }
        
        self.algorithm_patterns = {
            # Sorting algorithms
            r'sort\s*\(': ComplexityType.LINEARITHMIC,
            r'stable_sort\s*\(': ComplexityType.LINEARITHMIC,
            r'partial_sort\s*\(': ComplexityType.LINEARITHMIC,
            r'nth_element\s*\(': ComplexityType.LINEAR,
            r'make_heap\s*\(': ComplexityType.LINEAR,
            r'sort_heap\s*\(': ComplexityType.LINEARITHMIC,
            
            # Search algorithms
            r'binary_search\s*\(': ComplexityType.LOGARITHMIC,
            r'lower_bound\s*\(': ComplexityType.LOGARITHMIC,
            r'upper_bound\s*\(': ComplexityType.LOGARITHMIC,
            r'equal_range\s*\(': ComplexityType.LOGARITHMIC,
            r'find\s*\(': ComplexityType.LINEAR,
            r'search\s*\(': ComplexityType.LINEAR,
            
            # Set operations
            r'set_union\s*\(': ComplexityType.LINEAR,
            r'set_intersection\s*\(': ComplexityType.LINEAR,
            r'set_difference\s*\(': ComplexityType.LINEAR,
            r'set_symmetric_difference\s*\(': ComplexityType.LINEAR,
            r'merge\s*\(': ComplexityType.LINEAR,
            r'unique\s*\(': ComplexityType.LINEAR,
            
            # Numeric algorithms
            r'accumulate\s*\(': ComplexityType.LINEAR,
            r'inner_product\s*\(': ComplexityType.LINEAR,
            r'partial_sum\s*\(': ComplexityType.LINEAR,
            r'adjacent_difference\s*\(': ComplexityType.LINEAR,
            
            # Permutation algorithms
            r'next_permutation\s*\(': ComplexityType.LINEAR,
            r'prev_permutation\s*\(': ComplexityType.LINEAR,
            
            # Transform algorithms
            r'transform\s*\(': ComplexityType.LINEAR,
            r'for_each\s*\(': ComplexityType.LINEAR,
            
            # Min/Max algorithms
            r'min_element\s*\(': ComplexityType.LINEAR,
            r'max_element\s*\(': ComplexityType.LINEAR,
            r'minmax_element\s*\(': ComplexityType.LINEAR,
            
            # Modifying sequence operations
            r'copy\s*\(': ComplexityType.LINEAR,
            r'copy_if\s*\(': ComplexityType.LINEAR,
            r'move\s*\(': ComplexityType.LINEAR,
            r'fill\s*\(': ComplexityType.LINEAR,
            r'generate\s*\(': ComplexityType.LINEAR,
            r'replace\s*\(': ComplexityType.LINEAR,
            r'reverse\s*\(': ComplexityType.LINEAR,
            r'rotate\s*\(': ComplexityType.LINEAR,
            r'shuffle\s*\(': ComplexityType.LINEAR,
            
            # Special algorithms with higher complexity
            r'pow\s*\(': ComplexityType.LOGARITHMIC,  # Fast exponentiation
            r'gcd\s*\(': ComplexityType.LOGARITHMIC,  # Euclidean algorithm
            r'lcm\s*\(': ComplexityType.LOGARITHMIC,
        }
        
        self.data_structure_patterns = {
            # Tree-based structures (O(log n) operations)
            r'map\s*<': ComplexityType.LOGARITHMIC,
            r'set\s*<': ComplexityType.LOGARITHMIC,
            r'multimap\s*<': ComplexityType.LOGARITHMIC,
            r'multiset\s*<': ComplexityType.LOGARITHMIC,
            r'priority_queue\s*<': ComplexityType.LOGARITHMIC,
            
            # Hash-based structures (O(1) average operations)
            r'unordered_map\s*<': ComplexityType.CONSTANT,
            r'unordered_set\s*<': ComplexityType.CONSTANT,
            r'unordered_multimap\s*<': ComplexityType.CONSTANT,
            r'unordered_multiset\s*<': ComplexityType.CONSTANT,
            
            # Linear structures
            r'vector\s*<': ComplexityType.CONSTANT,  # Access is O(1), but operations vary
            r'list\s*<': ComplexityType.CONSTANT,    # Insert/delete O(1), search O(n)
            r'forward_list\s*<': ComplexityType.CONSTANT,
            r'deque\s*<': ComplexityType.CONSTANT,
            r'array\s*<': ComplexityType.CONSTANT,
            r'string\s*': ComplexityType.CONSTANT,
            
            # Stack and Queue
            r'stack\s*<': ComplexityType.CONSTANT,
            r'queue\s*<': ComplexityType.CONSTANT,
        }
        
        # Graph algorithm patterns
        self.graph_patterns = {
            r'dfs\s*\(|depth.*first': ComplexityType.V_PLUS_E,  # O(V + E)
            r'bfs\s*\(|breadth.*first': ComplexityType.V_PLUS_E,  # O(V + E)
            r'dijkstra': ComplexityType.V_LOG_V_PLUS_E,  # O((V + E) log V)
            r'bellman.*ford': ComplexityType.V_E,  # O(VE)
            r'floyd.*warshall': ComplexityType.V_CUBED,  # O(V³)
            r'kruskal': ComplexityType.E_LOG_V,  # O(E log E)
            r'prim': ComplexityType.E_LOG_V,  # O(E log V)
            r'topological.*sort': ComplexityType.V_PLUS_E,  # O(V + E)
            r'strongly.*connected': ComplexityType.V_PLUS_E,  # O(V + E)
            r'minimum.*spanning.*tree|mst': ComplexityType.E_LOG_V,  # O(E log V)
            r'shortest.*path': ComplexityType.V_LOG_V_PLUS_E,  # Usually Dijkstra
            r'all.*pairs.*shortest': ComplexityType.V_CUBED,  # Floyd-Warshall
            r'maximum.*flow|max.*flow': ComplexityType.V_E,  # O(VE²) or better
            r'minimum.*cut|min.*cut': ComplexityType.V_E,  # Similar to max flow
            r'bipartite.*matching': ComplexityType.E_SQRT_V,  # Hopcroft-Karp
            r'ford.*fulkerson': ComplexityType.V_E,  # O(VE²) worst case
            r'edmonds.*karp': ComplexityType.V_E,  # O(VE²)
            r'dinic': ComplexityType.V_SQUARED,  # O(V²E)
            r'articulation.*point|bridge.*finding': ComplexityType.V_PLUS_E,
            r'tarjan|kosaraju': ComplexityType.V_PLUS_E,  # SCC algorithms
        }
        
        # String algorithm patterns
        self.string_patterns = {
            r'kmp|knuth.*morris.*pratt': ComplexityType.TEXT_PLUS_PATTERN,
            r'rabin.*karp': ComplexityType.TEXT_PLUS_PATTERN,
            r'boyer.*moore': ComplexityType.TEXT_PLUS_PATTERN,
            r'z.*algorithm': ComplexityType.TEXT_PLUS_PATTERN,
            r'suffix.*array': ComplexityType.SUFFIX_ARRAY_CONSTRUCTION,
            r'suffix.*tree': ComplexityType.SUFFIX_TREE_CONSTRUCTION,
            r'lcp.*array': ComplexityType.LINEAR,  # Longest Common Prefix
            r'edit.*distance|levenshtein': ComplexityType.EDIT_DISTANCE,
            r'longest.*common.*subsequence|lcs': ComplexityType.LCS_DP,
            r'longest.*increasing.*subsequence|lis': ComplexityType.LINEARITHMIC,
        }
        
        # Mathematical algorithm patterns
        self.math_patterns = {
            r'sieve.*eratosthenes': ComplexityType.SIEVE_OF_ERATOSTHENES,  # O(n log log n)
            r'prime.*factorization': ComplexityType.PRIME_FACTORIZATION,  # O(√n)
            r'matrix.*multiplication': ComplexityType.CUBIC,  # O(n³) naive
            r'fft|fast.*fourier': ComplexityType.LINEARITHMIC,  # O(n log n)
            r'convolution': ComplexityType.LINEARITHMIC,
            r'segment.*tree': ComplexityType.SEGMENT_TREE_OPERATIONS,  # Operations O(log n)
            r'fenwick.*tree|binary.*indexed': ComplexityType.FENWICK_TREE_OPERATIONS,
            r'union.*find|disjoint.*set': ComplexityType.INVERSE_ACKERMANN,  # O(α(n)) ≈ O(log* n)
            r'trie': ComplexityType.TRIE_OPERATIONS,  # O(m) where m is string length
        }
        
        # Dynamic programming patterns
        self.dp_patterns = {
            r'dp\[.*\]\[.*\]': ComplexityType.QUADRATIC,  # 2D DP
            r'dp\[.*\]\[.*\]\[.*\]': ComplexityType.CUBIC,  # 3D DP
            r'memoization|memo': ComplexityType.LINEAR,  # Usually improves to polynomial
        }

        # Geometric algorithm patterns
        self.dp_patterns = {
            r'dp\[.*\]\[.*\]': ComplexityType.QUADRATIC,  # 2D DP
            r'dp\[.*\]\[.*\]\[.*\]': ComplexityType.CUBIC,  # 3D DP
            r'memoization|memo': ComplexityType.LINEAR,  # Usually improves to polynomial
        }

        # Geometric algorithm patterns
        self.geometric_patterns = {
            r'convex.*hull': ComplexityType.CONVEX_HULL_2D,  # O(n log n)
            r'closest.*pair': ComplexityType.CLOSEST_PAIR,  # O(n log n)
            r'line.*segment.*intersection': ComplexityType.LINE_INTERSECTION,
            r'voronoi.*diagram': ComplexityType.VORONOI_DIAGRAM,
            r'delaunay.*triangulation': ComplexityType.DELAUNAY_TRIANGULATION,
            r'point.*in.*polygon': ComplexityType.LINEAR,  # O(n) for simple polygon
            r'polygon.*area': ComplexityType.LINEAR,  # O(n)
            r'polygon.*perimeter': ComplexityType.LINEAR,  # O(n)
            r'bentley.*ottmann': ComplexityType.LINEARITHMIC,  # Sweep line
            r'rotating.*calipers': ComplexityType.LINEAR,  # O(n) for convex polygon
        }
        
        # Advanced sorting patterns
        self.sorting_patterns = {
            r'counting.*sort': ComplexityType.COUNTING_SORT,  # O(n + k)
            r'radix.*sort': ComplexityType.RADIX_SORT,  # O(d(n + k))
            r'bucket.*sort': ComplexityType.BUCKET_SORT,  # O(n + k)
            r'heap.*sort': ComplexityType.LINEARITHMIC,  # O(n log n)
            r'merge.*sort': ComplexityType.LINEARITHMIC,  # O(n log n)
            r'quick.*sort': ComplexityType.LINEARITHMIC,  # O(n log n) average
            r'intro.*sort': ComplexityType.LINEARITHMIC,  # O(n log n)
            r'tim.*sort': ComplexityType.LINEARITHMIC,  # O(n log n)
            r'shell.*sort': ComplexityType.N_TO_1_5,  # O(n^1.5) average
            r'insertion.*sort': ComplexityType.QUADRATIC,  # O(n²)
            r'selection.*sort': ComplexityType.QUADRATIC,  # O(n²)
            r'bubble.*sort': ComplexityType.QUADRATIC,  # O(n²)
        }
        
        # Advanced data structure patterns
        self.advanced_ds_patterns = {
            r'red.*black.*tree|rb.*tree': ComplexityType.AVL_OPERATIONS,  # O(log n)
            r'avl.*tree': ComplexityType.AVL_OPERATIONS,  # O(log n)
            r'splay.*tree': ComplexityType.LOGARITHMIC,  # O(log n) amortized
            r'b.*tree|b\+.*tree': ComplexityType.LOGARITHMIC,  # O(log n)
            r'skip.*list': ComplexityType.LOGARITHMIC,  # O(log n) expected
            r'bloom.*filter': ComplexityType.CONSTANT,  # O(1) for operations
            r'hash.*table.*chaining': ComplexityType.CONSTANT,  # O(1) average
            r'hash.*table.*probing': ComplexityType.CONSTANT,  # O(1) average
            r'disjoint.*set.*union': ComplexityType.INVERSE_ACKERMANN,  # O(α(n))
            r'persistent.*data.*structure': ComplexityType.LOGARITHMIC,  # Usually O(log n)
        }
        
    def extract_functions(self, code: str) -> List[Tuple[str, str, int]]:
        """Extract function definitions from C++ code."""
        functions = []
        
        # Pattern to match function definitions
        function_pattern = re.compile(
            r'(?:(?:inline|static|virtual|explicit|const|constexpr|noexcept)\s+)*'
            r'(?:(?:unsigned|signed|long|short|const)\s+)*'
            r'(?:\w+(?:\s*::\s*\w+)*(?:\s*<[^>]*>)?(?:\s*\*|\s*&)*\s+)'  # return type
            r'(\w+)\s*'  # function name
            r'\([^)]*\)\s*'  # parameters
            r'(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?'
            r'\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',  # function body
            re.MULTILINE | re.DOTALL
        )
        
        lines = code.split('\n')
        for match in function_pattern.finditer(code):
            func_name = match.group(1)
            func_body = match.group(2)
            
            # Find line number
            line_num = code[:match.start()].count('\n') + 1
            
            # Skip constructors, destructors, and operators
            if not (func_name.startswith('~') or func_name == 'operator' or 
                   func_name[0].isupper()):
                functions.append((func_name, func_body, line_num))
        
        return functions

    def analyze_loops(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze loop structures in the code."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 1.0
        
        # Count nested loops
        nested_level = self._count_nested_loops(code)
        if nested_level > 0:
            reasons.append(f"Found {nested_level} levels of nested loops")
            if nested_level == 1:
                max_complexity = ComplexityType.LINEAR
            elif nested_level == 2:
                max_complexity = ComplexityType.QUADRATIC
            elif nested_level == 3:
                max_complexity = ComplexityType.CUBIC
            elif nested_level == 4:
                max_complexity = ComplexityType.QUARTIC
            else:
                max_complexity = ComplexityType.POLYNOMIAL
                reasons.append(f"Polynomial complexity with degree {nested_level}")
        
        # Check for specific loop patterns
        for pattern, complexity in self.loop_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                reasons.append(f"Found loop pattern suggesting {complexity.value}")
                if self._complexity_order(complexity) > self._complexity_order(max_complexity):
                    max_complexity = complexity
        
        # Check for divide-and-conquer patterns
        if re.search(r'(\w+)\s*\(\s*[^,)]*,\s*[^,)]*\+\s*[^,)]*\s*\/\s*2', code):
            reasons.append("Found divide-and-conquer pattern")
            if max_complexity == ComplexityType.LINEAR:
                max_complexity = ComplexityType.LINEARITHMIC
            elif max_complexity == ComplexityType.CONSTANT:
                max_complexity = ComplexityType.LOGARITHMIC
        
        return max_complexity, reasons, confidence

    def analyze_algorithms(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze standard algorithm usage."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.9
        
        for pattern, complexity in self.algorithm_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                algo_name = pattern.replace(r'\s*\(', '').replace(r'\\', '')
                reasons.append(f"Uses {algo_name} algorithm ({complexity.value})")
                if self._complexity_order(complexity) > self._complexity_order(max_complexity):
                    max_complexity = complexity
        
        return max_complexity, reasons, confidence

    def analyze_data_structures(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze data structure usage patterns."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.8
        
        for pattern, complexity in self.data_structure_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                ds_name = pattern.replace(r'\s*<', '').replace(r'\\', '')
                reasons.append(f"Uses {ds_name} data structure")
                # Data structure complexity affects operations
                if complexity == ComplexityType.LOGARITHMIC and max_complexity == ComplexityType.CONSTANT:
                    max_complexity = ComplexityType.LOGARITHMIC
        
        return max_complexity, reasons, confidence

    def analyze_recursion(self, code: str, function_name: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze recursive patterns."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.7
        
        # Check if function calls itself
        if re.search(rf'\b{function_name}\s*\(', code):
            reasons.append("Function is recursive")
            
            # Count number of recursive calls in the function
            recursive_calls = len(re.findall(rf'\b{function_name}\s*\(', code))
            
            if recursive_calls == 1:
                # Single recursive call - analyze the reduction pattern
                if re.search(r'n\s*[-*/]\s*2|n\s*\/\s*2|n\s*>>\s*1|n\s*\/\s*2', code):
                    max_complexity = ComplexityType.LOGARITHMIC
                    reasons.append("Recursive calls with input halving (binary search/divide-conquer)")
                elif re.search(r'n\s*-\s*1|n\s*--|\w+\s*-\s*1', code):
                    max_complexity = ComplexityType.LINEAR
                    reasons.append("Linear recursion (n-1 pattern)")
                elif re.search(r'sqrt\s*\(\s*n\s*\)', code):
                    max_complexity = ComplexityType.SQRT
                    reasons.append("Recursive calls with square root reduction")
                else:
                    max_complexity = ComplexityType.LINEAR
                    reasons.append("Single recursive call per invocation")
                    
            elif recursive_calls == 2:
                # Two recursive calls - check for divide and conquer
                if re.search(r'n\s*/\s*2|n\s*>>\s*1', code):
                    max_complexity = ComplexityType.LINEARITHMIC
                    reasons.append("Divide-and-conquer with two recursive calls (merge sort pattern)")
                else:
                    max_complexity = ComplexityType.EXPONENTIAL_BASE_2
                    reasons.append("Two recursive calls per invocation (likely exponential)")
                    
            elif recursive_calls == 3:
                max_complexity = ComplexityType.EXPONENTIAL_BASE_3
                reasons.append("Three recursive calls per invocation")
                
            elif recursive_calls > 3:
                max_complexity = ComplexityType.EXPONENTIAL_BASE_K
                reasons.append(f"{recursive_calls} recursive calls per invocation")
                
            # Check for memoization or dynamic programming
            if re.search(r'memo|cache|dp\[', code):
                reasons.append("Memoization detected - complexity may be improved")
                if max_complexity == ComplexityType.EXPONENTIAL_BASE_2:
                    max_complexity = ComplexityType.QUADRATIC
                elif max_complexity == ComplexityType.EXPONENTIAL_BASE_3:
                    max_complexity = ComplexityType.CUBIC
                confidence = 0.8
        
        return max_complexity, reasons, confidence

    def _count_nested_loops(self, code: str) -> int:
        """Count the maximum nesting level of loops."""
        max_nesting = 0
        current_nesting = 0
        
        # Remove strings and comments to avoid false positives
        code_cleaned = self._remove_strings_and_comments(code)
        
        i = 0
        while i < len(code_cleaned):
            # Check for loop keywords
            if (code_cleaned[i:i+3] == 'for' or 
                code_cleaned[i:i+5] == 'while'):
                # Find the opening brace
                j = i
                while j < len(code_cleaned) and code_cleaned[j] != '{':
                    j += 1
                if j < len(code_cleaned):
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                i = j
            elif code_cleaned[i] == '}':
                if current_nesting > 0:
                    current_nesting -= 1
            i += 1
        
        return max_nesting

    def _remove_strings_and_comments(self, code: str) -> str:
        """Remove string literals and comments from code."""
        # Remove single-line comments
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Remove string literals
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)
        return code

    def _complexity_order(self, complexity: ComplexityType) -> int:
        """Return the order of complexity for comparison."""
        order_map = {
            # Constant and sub-linear
            ComplexityType.CONSTANT: 0,
            ComplexityType.AMORTIZED_CONSTANT: 1,
            ComplexityType.INVERSE_ACKERMANN: 2,
            ComplexityType.LOG_STAR: 3,
            ComplexityType.LOG_LOG: 4,
            ComplexityType.GCD_EUCLIDEAN: 5,
            ComplexityType.MODULAR_EXPONENTIATION: 6,
            ComplexityType.LOGARITHMIC: 7,
            ComplexityType.AVL_OPERATIONS: 8,
            ComplexityType.HEAP_OPERATIONS: 9,
            ComplexityType.SEGMENT_TREE_OPERATIONS: 10,
            ComplexityType.FENWICK_TREE_OPERATIONS: 11,
            ComplexityType.LOG_N_OVER_LOG_LOG_N: 12,
            ComplexityType.LOG_SQUARED: 13,
            ComplexityType.LOG_CUBED: 14,
            
            # Sub-linear to linear
            ComplexityType.N_TO_2_3: 15,
            ComplexityType.N_TO_3_4: 16,
            ComplexityType.N_TO_4_5: 17,
            ComplexityType.N_TO_5_6: 18,
            ComplexityType.SQRT: 19,
            ComplexityType.PRIME_FACTORIZATION: 20,
            ComplexityType.SQRT_LOG: 21,
            
            # Linear and near-linear
            ComplexityType.LINEAR: 22,
            ComplexityType.EXPECTED_LINEAR: 23,
            ComplexityType.V_PLUS_E: 24,
            ComplexityType.STRING_LENGTH: 25,
            ComplexityType.TEXT_PLUS_PATTERN: 26,
            ComplexityType.SUFFIX_TREE_CONSTRUCTION: 27,
            ComplexityType.TRIE_OPERATIONS: 28,
            ComplexityType.N_LOG_LOG: 29,
            ComplexityType.SIEVE_OF_ERATOSTHENES: 30,
            ComplexityType.LINEARITHMIC: 31,
            ComplexityType.COMPARISON_SORT_OPTIMAL: 32,
            ComplexityType.SUFFIX_ARRAY_CONSTRUCTION: 33,
            ComplexityType.CONVEX_HULL_2D: 34,
            ComplexityType.CLOSEST_PAIR: 35,
            ComplexityType.LINE_INTERSECTION: 36,
            ComplexityType.VORONOI_DIAGRAM: 37,
            ComplexityType.DELAUNAY_TRIANGULATION: 38,
            ComplexityType.E_LOG_V: 39,
            ComplexityType.N_LOG_SQUARED: 40,
            ComplexityType.N_LOG_CUBED: 41,
            ComplexityType.V_LOG_V_PLUS_E: 42,
            ComplexityType.V_LOG_V_PLUS_E_ALPHA: 43,
            ComplexityType.N_TO_1_5: 44,
            ComplexityType.N_SQRT: 45,
            ComplexityType.N_SQRT_LOG: 46,
            
            # Quadratic and related
            ComplexityType.N_TO_5_3: 47,
            ComplexityType.N_TO_7_4: 48,
            ComplexityType.QUADRATIC: 49,
            ComplexityType.V_SQUARED: 50,
            ComplexityType.TEXT_TIMES_PATTERN: 51,
            ComplexityType.LCS_DP: 52,
            ComplexityType.EDIT_DISTANCE: 53,
            ComplexityType.N_SQUARED_LOG_LOG: 54,
            ComplexityType.N_SQUARED_LOG: 55,
            ComplexityType.N_SQUARED_LOG_SQUARED: 56,
            ComplexityType.N_TO_2_5: 57,
            ComplexityType.MATRIX_MULT_OPTIMAL: 58,
            ComplexityType.MATRIX_MULT_COPPERSMITH: 59,
            ComplexityType.N_TO_8_3: 60,
            ComplexityType.MATRIX_MULT_STRASSEN: 61,
            
            # Cubic and higher polynomials
            ComplexityType.CUBIC: 62,
            ComplexityType.MATRIX_MULT_NAIVE: 63,
            ComplexityType.V_E: 64,
            ComplexityType.V_CUBED: 65,
            ComplexityType.N_CUBED_LOG: 66,
            ComplexityType.N_TO_3_5: 67,
            ComplexityType.QUARTIC: 68,
            ComplexityType.N_TO_4_5_ALT: 69,
            ComplexityType.QUINTIC: 70,
            ComplexityType.SEXTIC: 71,
            ComplexityType.POLYNOMIAL: 72,
            
            # Special data structure complexities
            ComplexityType.COUNTING_SORT: 73,
            ComplexityType.RADIX_SORT: 74,
            ComplexityType.BUCKET_SORT: 75,
            ComplexityType.HASH_TABLE_WORST: 76,
            ComplexityType.BST_WORST: 77,
            ComplexityType.E_SQRT_V: 78,
            
            # Exponential and super-exponential
            ComplexityType.EXPONENTIAL_BASE_2: 79,
            ComplexityType.EXPONENTIAL_BASE_3: 80,
            ComplexityType.EXPONENTIAL_BASE_E: 81,
            ComplexityType.EXPONENTIAL_BASE_K: 82,
            ComplexityType.EXPONENTIAL_LINEAR: 83,
            ComplexityType.EXPONENTIAL_QUADRATIC: 84,
            ComplexityType.EXPONENTIAL_POLY: 85,
            
            # Factorial and super-exponential
            ComplexityType.FACTORIAL: 86,
            ComplexityType.FACTORIAL_APPROX: 87,
            ComplexityType.DOUBLE_EXPONENTIAL: 88,
            ComplexityType.TOWER_FUNCTION: 89,
            ComplexityType.ACKERMANN: 90,
            
            # Special cases
            ComplexityType.PROBABILISTIC: 91,
            ComplexityType.UNKNOWN: 100
        }
        return order_map.get(complexity, 100)

    def analyze_function(self, function_name: str, function_body: str, line_number: int) -> ComplexityResult:
        """Analyze a single function's time complexity."""
        all_reasons = []
        complexities = []
        confidences = []
        
        # Analyze different aspects
        loop_complexity, loop_reasons, loop_confidence = self.analyze_loops(function_body)
        if loop_complexity != ComplexityType.CONSTANT:
            complexities.append(loop_complexity)
            confidences.append(loop_confidence)
            all_reasons.extend(loop_reasons)
        
        algo_complexity, algo_reasons, algo_confidence = self.analyze_algorithms(function_body)
        if algo_complexity != ComplexityType.CONSTANT:
            complexities.append(algo_complexity)
            confidences.append(algo_confidence)
            all_reasons.extend(algo_reasons)
        
        ds_complexity, ds_reasons, ds_confidence = self.analyze_data_structures(function_body)
        if ds_complexity != ComplexityType.CONSTANT:
            complexities.append(ds_complexity)
            confidences.append(ds_confidence)
            all_reasons.extend(ds_reasons)
        
        rec_complexity, rec_reasons, rec_confidence = self.analyze_recursion(function_body, function_name)
        if rec_complexity != ComplexityType.CONSTANT:
            complexities.append(rec_complexity)
            confidences.append(rec_confidence)
            all_reasons.extend(rec_reasons)
        
        # Analyze graph algorithms
        graph_complexity, graph_reasons, graph_confidence = self.analyze_graph_algorithms(function_body)
        if graph_complexity != ComplexityType.CONSTANT:
            complexities.append(graph_complexity)
            confidences.append(graph_confidence)
            all_reasons.extend(graph_reasons)
        
        # Analyze string algorithms
        string_complexity, string_reasons, string_confidence = self.analyze_string_algorithms(function_body)
        if string_complexity != ComplexityType.CONSTANT:
            complexities.append(string_complexity)
            confidences.append(string_confidence)
            all_reasons.extend(string_reasons)
        
        # Analyze mathematical algorithms
        math_complexity, math_reasons, math_confidence = self.analyze_math_algorithms(function_body)
        if math_complexity != ComplexityType.CONSTANT:
            complexities.append(math_complexity)
            confidences.append(math_confidence)
            all_reasons.extend(math_reasons)
        
        # Analyze dynamic programming patterns
        dp_complexity, dp_reasons, dp_confidence = self.analyze_dp_patterns(function_body)
        if dp_complexity != ComplexityType.CONSTANT:
            complexities.append(dp_complexity)
            confidences.append(dp_confidence)
            all_reasons.extend(dp_reasons)
        
        # Analyze special patterns
        special_complexity, special_reasons, special_confidence = self.analyze_special_patterns(function_body)
        if special_complexity != ComplexityType.CONSTANT:
            complexities.append(special_complexity)
            confidences.append(special_confidence)
            all_reasons.extend(special_reasons)
        
        # Analyze geometric algorithms
        geometric_complexity, geometric_reasons, geometric_confidence = self.analyze_geometric_algorithms(function_body)
        if geometric_complexity != ComplexityType.CONSTANT:
            complexities.append(geometric_complexity)
            confidences.append(geometric_confidence)
            all_reasons.extend(geometric_reasons)
        
        # Analyze sorting algorithms
        sorting_complexity, sorting_reasons, sorting_confidence = self.analyze_sorting_algorithms(function_body)
        if sorting_complexity != ComplexityType.CONSTANT:
            complexities.append(sorting_complexity)
            confidences.append(sorting_confidence)
            all_reasons.extend(sorting_reasons)
        
        # Analyze advanced data structures
        advanced_ds_complexity, advanced_ds_reasons, advanced_ds_confidence = self.analyze_advanced_data_structures(function_body)
        if advanced_ds_complexity != ComplexityType.CONSTANT:
            complexities.append(advanced_ds_complexity)
            confidences.append(advanced_ds_confidence)
            all_reasons.extend(advanced_ds_reasons)
        
        # Determine final complexity
        if not complexities:
            final_complexity = ComplexityType.CONSTANT
            final_confidence = 1.0
            all_reasons.append("No complex operations detected")
        else:
            # Take the highest complexity
            final_complexity = max(complexities, key=self._complexity_order)
            # Average confidence weighted by complexity order
            final_confidence = sum(confidences) / len(confidences)
        
        return ComplexityResult(
            function_name=function_name,
            complexity=final_complexity,
            confidence=final_confidence,
            reasons=all_reasons,
            line_number=line_number
        )

    def analyze_file(self, file_path: str) -> List[ComplexityResult]:
        """Analyze a C++ file and return complexity results for all functions."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    code = f.read()
            except Exception as e:
                logger.error(f"Could not read file {file_path}: {e}")
                return []
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        functions = self.extract_functions(code)
        results = []
        
        for func_name, func_body, line_num in functions:
            result = self.analyze_function(func_name, func_body, line_num)
            results.append(result)
        
        return results

    def analyze_graph_algorithms(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze graph algorithm patterns."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.85
        
        for pattern, complexity in self.graph_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                algo_name = pattern.replace(r'.*', ' ').replace(r'\\', '')
                reasons.append(f"Uses graph algorithm: {algo_name} ({complexity.value})")
                if self._complexity_order(complexity) > self._complexity_order(max_complexity):
                    max_complexity = complexity
        
        return max_complexity, reasons, confidence

    def analyze_string_algorithms(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze string algorithm patterns."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.85
        
        for pattern, complexity in self.string_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                algo_name = pattern.replace(r'.*', ' ').replace(r'\\', '')
                reasons.append(f"Uses string algorithm: {algo_name} ({complexity.value})")
                if self._complexity_order(complexity) > self._complexity_order(max_complexity):
                    max_complexity = complexity
        
        return max_complexity, reasons, confidence

    def analyze_math_algorithms(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze mathematical algorithm patterns."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.85
        
        for pattern, complexity in self.math_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                algo_name = pattern.replace(r'.*', ' ').replace(r'\\', '')
                reasons.append(f"Uses mathematical algorithm: {algo_name} ({complexity.value})")
                if self._complexity_order(complexity) > self._complexity_order(max_complexity):
                    max_complexity = complexity
        
        return max_complexity, reasons, confidence

    def analyze_dp_patterns(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze dynamic programming patterns."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.8
        
        for pattern, complexity in self.dp_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                dp_type = pattern.replace(r'.*', ' ').replace(r'\\', '')
                reasons.append(f"Uses dynamic programming: {dp_type} ({complexity.value})")
                if self._complexity_order(complexity) > self._complexity_order(max_complexity):
                    max_complexity = complexity
        
        return max_complexity, reasons, confidence

    def analyze_special_patterns(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze special complexity patterns."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.7
        
        # Check for factorial patterns
        if re.search(r'factorial|permutation.*all|generate.*all.*permutation', code, re.IGNORECASE):
            max_complexity = ComplexityType.FACTORIAL
            reasons.append("Factorial pattern detected (generating all permutations)")
            
        # Check for exponential patterns
        elif re.search(r'subset.*all|powerset|generate.*all.*subset', code, re.IGNORECASE):
            max_complexity = ComplexityType.EXPONENTIAL_BASE_2
            reasons.append("Exponential pattern detected (generating all subsets)")
            
        # Check for double exponential patterns
        elif re.search(r'tower.*hanoi|ackermann', code, re.IGNORECASE):
            max_complexity = ComplexityType.DOUBLE_EXPONENTIAL
            reasons.append("Double exponential pattern detected")
            
        # Check for n^n patterns
        elif re.search(r'n.*\*.*n.*\*.*n|pow.*n.*n', code, re.IGNORECASE):
            max_complexity = ComplexityType.EXPONENTIAL_POLY
            reasons.append("n^n pattern detected")
        
        return max_complexity, reasons, confidence

    def analyze_geometric_algorithms(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze geometric algorithm patterns."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.85
        
        for pattern, complexity in self.geometric_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                algo_name = pattern.replace(r'.*', ' ').replace(r'\\', '')
                reasons.append(f"Uses geometric algorithm: {algo_name} ({complexity.value})")
                if self._complexity_order(complexity) > self._complexity_order(max_complexity):
                    max_complexity = complexity
        
        return max_complexity, reasons, confidence

    def analyze_sorting_algorithms(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze sorting algorithm patterns."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.9
        
        for pattern, complexity in self.sorting_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                algo_name = pattern.replace(r'.*', ' ').replace(r'\\', '')
                reasons.append(f"Uses sorting algorithm: {algo_name} ({complexity.value})")
                if self._complexity_order(complexity) > self._complexity_order(max_complexity):
                    max_complexity = complexity
        
        return max_complexity, reasons, confidence

    def analyze_advanced_data_structures(self, code: str) -> Tuple[ComplexityType, List[str], float]:
        """Analyze advanced data structure patterns."""
        reasons = []
        max_complexity = ComplexityType.CONSTANT
        confidence = 0.85
        
        for pattern, complexity in self.advanced_ds_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                ds_name = pattern.replace(r'.*', ' ').replace(r'\\', '')
                reasons.append(f"Uses advanced data structure: {ds_name} ({complexity.value})")
                if self._complexity_order(complexity) > self._complexity_order(max_complexity):
                    max_complexity = complexity
        
        return max_complexity, reasons, confidence
def main():
    parser = argparse.ArgumentParser(description='Analyze time complexity of C++ code')
    parser.add_argument('path', help='Path to C++ file or directory')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='Recursively analyze all C++ files in directory')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Show detailed analysis reasons')
    parser.add_argument('-o', '--output', help='Output file for results')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                       help='Minimum confidence threshold for results')
    
    args = parser.parse_args()
    
    analyzer = CppTimeComplexityAnalyzer()
    all_results = []
    
    path = Path(args.path)
    
    if path.is_file():
        if path.suffix in ['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h']:
            results = analyzer.analyze_file(str(path))
            all_results.extend([(str(path), results)])
        else:
            logger.error(f"File {path} is not a C++ source file")
            return 1
    elif path.is_dir():
        cpp_extensions = ['*.cpp', '*.cc', '*.cxx', '*.c++', '*.hpp', '*.h']
        cpp_files = []
        
        if args.recursive:
            for ext in cpp_extensions:
                cpp_files.extend(path.rglob(ext))
        else:
            for ext in cpp_extensions:
                cpp_files.extend(path.glob(ext))
        
        if not cpp_files:
            logger.error(f"No C++ files found in {path}")
            return 1
        
        for cpp_file in cpp_files:
            results = analyzer.analyze_file(str(cpp_file))
            if results:
                all_results.extend([(str(cpp_file), results)])
    else:
        logger.error(f"Path {path} does not exist")
        return 1
    
    # Filter by confidence threshold
    if args.min_confidence > 0:
        filtered_results = []
        for file_path, results in all_results:
            filtered = [r for r in results if r.confidence >= args.min_confidence]
            if filtered:
                filtered_results.append((file_path, filtered))
        all_results = filtered_results
    
    # Display results
    if args.output:
        with open(args.output, 'w') as f:
            f.write("Time Complexity Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            for file_path, results in all_results:
                f.write(f"File: {file_path}\n")
                f.write("-" * 40 + "\n")
                
                for result in results:
                    f.write(f"Function: {result.function_name} (line {result.line_number})\n")
                    f.write(f"Complexity: {result.complexity.value}\n")
                    f.write(f"Confidence: {result.confidence:.2f}\n")
                    
                    if args.verbose and result.reasons:
                        f.write("Reasons:\n")
                        for reason in result.reasons:
                            f.write(f"  - {reason}\n")
                    f.write("\n")
                f.write("\n")
    else:
        print("Time Complexity Analysis Results")
        print("=" * 50)
        
        for file_path, results in all_results:
            print(f"\nFile: {file_path}")
            print("-" * 40)
            
            for result in results:
                print(f"Function: {result.function_name} (line {result.line_number})")
                print(f"Complexity: {result.complexity.value}")
                print(f"Confidence: {result.confidence:.2f}")
                
                if args.verbose and result.reasons:
                    print("Reasons:")
                    for reason in result.reasons:
                        print(f"  - {reason}")
                print()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
