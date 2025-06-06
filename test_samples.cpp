#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>

// O(1) - Constant time
int getFirst(const std::vector<int>& arr) {
    if (arr.empty()) return -1;
    return arr[0];
}

// O(n) - Linear time
int linearSearch(const std::vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

// O(log n) - Logarithmic time (binary search)
int binarySearch(const std::vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

// O(n²) - Quadratic time (bubble sort)
void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// O(n log n) - Using STL sort
void efficientSort(std::vector<int>& arr) {
    std::sort(arr.begin(), arr.end());
}

// O(log n) - Recursive binary search
int recursiveBinarySearch(const std::vector<int>& arr, int target, int left, int right) {
    if (left > right) {
        return -1;
    }
    
    int mid = left + (right - left) / 2;
    
    if (arr[mid] == target) {
        return mid;
    } else if (arr[mid] < target) {
        return recursiveBinarySearch(arr, target, mid + 1, right);
    } else {
        return recursiveBinarySearch(arr, target, left, mid - 1);
    }
}

// O(2^n) - Exponential time (naive fibonacci)
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// O(n) - Linear fibonacci with memoization concept
int fibonacciLinear(int n) {
    if (n <= 1) return n;
    
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

// O(n³) - Cubic time (naive matrix multiplication)
void matrixMultiply(const std::vector<std::vector<int>>& A, 
                   const std::vector<std::vector<int>>& B,
                   std::vector<std::vector<int>>& C) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// O(log n) - Using map operations
void mapOperations(std::map<int, std::string>& myMap, int key) {
    myMap[key] = "value";
    auto it = myMap.find(key);
    if (it != myMap.end()) {
        myMap.erase(it);
    }
}

int main() {
    std::vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    std::cout << "Testing time complexity analyzer with sample functions" << std::endl;
    
    // Test various functions
    std::cout << "First element: " << getFirst(arr) << std::endl;
    std::cout << "Linear search for 5: " << linearSearch(arr, 5) << std::endl;
    std::cout << "Binary search for 7: " << binarySearch(arr, 7) << std::endl;
    
    bubbleSort(arr);
    efficientSort(arr);
    
    std::cout << "Recursive binary search for 3: " 
              << recursiveBinarySearch(arr, 3, 0, arr.size() - 1) << std::endl;
    
    std::cout << "Fibonacci(10): " << fibonacci(10) << std::endl;
    std::cout << "Linear Fibonacci(10): " << fibonacciLinear(10) << std::endl;
    
    return 0;
}
