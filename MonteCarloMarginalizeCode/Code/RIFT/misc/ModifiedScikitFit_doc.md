# ModifiedScikitFit.py Documentation

## Overview

ModifiedScikitFit.py is a modified version of scikit-learn's preprocessing module that extends standard data preprocessing algorithms with additional functionality for handling symmetry in polynomial feature generation.

## Key Modifications

### 1. PolynomialFeatures Class Extension

The most significant modification is in the `PolynomialFeatures` class, which has been extended with symmetry handling capabilities:

#### New Parameters:
- `symmetry_list`: List of symmetry values for input features (default=None)

#### Modified Methods:
- `_combinations()`: Now filters combinations based on symmetry properties
- `powers_()`: Updated to work with symmetry-aware combinations
- `get_feature_names()`: Enhanced to work with symmetry constraints

#### Symmetry Handling Logic:
The modified `_combinations` method filters feature combinations based on their symmetry signature:
```python
# Select items with only +1 symmetry
combos_ok = []
for line in combos:
    signature = np.prod( np.array(self.symmetry_list)[list(line)] )
    if signature >=0:
        combos_ok.append(line)
```

This ensures only combinations with positive symmetry signatures are included in the polynomial feature generation.

## Original Functionality Retained

The file maintains all original scikit-learn preprocessing functionality:

### Data Scaling Classes:
- `MinMaxScaler`: Scale features to a given range
- `StandardScaler`: Standardize features by removing mean and scaling to unit variance
- `MaxAbsScaler`: Scale each feature by its maximum absolute value
- `RobustScaler`: Scale features using statistics robust to outliers

### Feature Transformation Classes:
- `PolynomialFeatures`: Generate polynomial and interaction features (with symmetry extension)
- `Normalizer`: Normalize samples individually to unit norm
- `Binarizer`: Binarize data according to a threshold
- `OneHotEncoder`: Encode categorical integer features using one-hot encoding
- `KernelCenterer`: Center a kernel matrix

### Utility Functions:
- `scale()`: Standardize a dataset along any axis
- `minmax_scale()`: Transform features by scaling to a given range
- `robust_scale()`: Standardize using median and interquartile range
- `normalize()`: Scale input vectors individually to unit norm
- `binarize()`: Boolean thresholding of array-like data

## Dependencies

- numpy
- scipy
- scikit-learn
- six

## Usage Examples

### Standard Usage (Unchanged)
```python
from ModifiedScikitFit import StandardScaler, MinMaxScaler

# Standard scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Min-max scaling
min_max_scaler = MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(data)
```

### Symmetry-Aware Polynomial Features
```python
from ModifiedScikitFit import PolynomialFeatures

# Create polynomial features with symmetry constraints
symmetry_list = [1, -1, 1]  # Symmetry properties for each feature
poly = PolynomialFeatures(degree=2, symmetry_list=symmetry_list)
poly_features = poly.fit_transform(data)
```

## Notes

- The modifications are backward compatible with standard scikit-learn preprocessing
- The symmetry functionality is only applicable to the `PolynomialFeatures` class
- All other functionality remains identical to the original scikit-learn implementation
- The file includes proper error handling and validation for the new symmetry features