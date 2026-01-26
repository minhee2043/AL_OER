"""
Gaussian Process Regression for Active Learning in Catalyst Discovery.

This module implements GPR with an integer-valued RBF kernel for predicting
adsorption energies on alloy surfaces. It includes active learning functionality
for iterative batch selection.

Usage:
    python mygaussian.py [batch_number]
    
Example:
    python mygaussian.py 15  # For final batch 15
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import sklearn.gaussian_process as gp
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.gaussian_process.kernels import (
    StationaryKernelMixin,
    NormalizedKernelMixin,
    Kernel,
    Hyperparameter,
    WhiteKernel
)
from scipy.spatial.distance import pdist, cdist, squareform


# ============================================================================
# Custom Kernel for Integer-Valued Variables
# ============================================================================

def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            f"Anisotropic kernel must have the same number of dimensions as data "
            f"({length_scale.shape[0]}!={X.shape[1]})"
        )
    return length_scale


class RBF_int(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X = np.around(X)  # Round to nearest integer
        length_scale = _check_length_scale(X, self.length_scale)
        
        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            else:
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        return K


# ============================================================================
# Data Loading and Model Creation
# ============================================================================

def load_and_preprocess_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")
    
    x_data, y_data = [], []
    with open(filename, 'r') as handle:
        for line_num, line in enumerate(handle.readlines(), 1):
            try:
                features = line.split(',')[:15]
                energy = float(line.split(',')[15])
                x_data.append(features)
                y_data.append(energy)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping line {line_num} in {filename}: {e}")
                continue
    
    X = np.array(x_data, dtype=int)
    y = np.array(y_data)
    
    print(f"✓ Loaded {len(X)} samples from {filename}")
    return X, y


def create_gpr_model(n_features):
    kernel = gp.kernels.ConstantKernel(1, (1e-1, 1e3)) * RBF_int(
        length_scale=0.2 * np.ones((n_features,)),
        length_scale_bounds=(1.0e-1, 1.0e3)
    )
    
    return gp.GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=0.05,
        normalize_y=True
    )


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance using MAE.
    """
    y_pred = model.predict(X_test, return_std=False)
    mae = mean_absolute_error(y_test, y_pred)
    return mae


# ============================================================================
# Active Learning and Batch Generation
# ============================================================================

def generate_batch_suggestions(model1, model2, batch_number,
                              dataspace_file='GPRdataspace.csv',
                              possible_file='possibleFp.csv',
                              index_file='index_metal.csv',
                              output_dir='./'):
    """
    Generate batch suggestions based on GPR predictions and acquisition function.
    
    This function:
    1. Predicts energies for all configurations
    2. Calculates acquisition scores (Expected Improvement)
    3. Ranks configurations
    4. Selects top candidates from DFT-compatible subset
    
    Args:
        model1 (GaussianProcessRegressor): Trained model for *O adsorption
        model2 (GaussianProcessRegressor): Trained model for *OH adsorption
        batch_number (int): Current batch iteration (e.g., 15)
        dataspace_file (str): Path to complete dataspace CSV
        possible_file (str): Path to DFT-compatible configurations CSV
        index_file (str): Path to metal index CSV
        output_dir (str): Directory to save output files
    
    Output Files:
        - GPR_batch{N}.csv: Predictions for all configurations
        - GPR_batch{N}_arrange.csv: Sorted by acquisition score
        - batch{N+1}_suggest.csv: Suggested surface features
        - batch{N+1}_metal.csv: Suggested metal sequences
    """
    print(f"\nGenerating batch {batch_number} suggestions...")
    
    output_file = os.path.join(output_dir, f'GPR_batch{batch_number}.csv')
    arranged_file = os.path.join(output_dir, f'GPR_batch{batch_number}_arrange.csv')
    suggest_file = os.path.join(output_dir, f'batch{batch_number+1}_suggest.csv')
    metal_file = os.path.join(output_dir, f'batch{batch_number+1}_metal.csv')
    
    print(f"  Loading dataspace from: {dataspace_file}")
    x_val = []
    multiplicities = []
    with open(dataspace_file, 'r') as handle:
        for line in handle.readlines():
            features = line.split(',')[:15]
            mult = int(line.split(',')[15])
            x_val.append(features)
            multiplicities.append(mult)
    
    X_val = np.array(x_val, dtype=int)
    print(f"   Loaded {len(X_val)} configurations")
    
    # Get predictions and uncertainties from both models
    print("  Predicting *O adsorption energies...")
    y_val1, dev1 = model1.predict(X_val, return_std=True)
    
    print("  Predicting *OH adsorption energies...")
    y_val2, dev2 = model2.predict(X_val, return_std=True)
    
    # Calculate acquisition score (Expected Improvement)
    # Target: |E_O - E_OH - 5.3| should be minimized
    E_target = 5.3
    diff_from_target = -np.abs((y_val1 - y_val2) - E_target)
    
    # Expected Improvement calculation
    avg_uncertainty = (dev1 + dev2) / 2
    Z = diff_from_target / avg_uncertainty
    acquisition_score = diff_from_target * norm.cdf(Z) + dev1 * norm.pdf(Z)
    
    output = np.c_[
        X_val,                    # Columns 0-14: Features
        multiplicities,           # Column 15: Multiplicity
        y_val1,                   # Column 16: E_O prediction
        y_val2,                   # Column 17: E_OH prediction
        y_val1 - y_val2,         # Column 18: E_O - E_OH
        dev1,                     # Column 19: Uncertainty O
        dev2,                     # Column 20: Uncertainty OH
        acquisition_score        # Column 21: Acquisition score
    ]
    
    np.savetxt(output_file, output, 
              fmt=['%d']*16 + ['%.5f']*6, 
              delimiter=',')
    print(f"   Saved predictions to: {output_file}")
    
    data = pd.read_csv(output_file, header=None)
    data.sort_values(data.columns[21], axis=0, ascending=False, inplace=True)
    data.to_csv(arranged_file, header=None, index=None, 
                columns=list(range(16)))  # Keep only features + multiplicity
    print(f"   Sorted configurations by acquisition score")
    
    generate_final_suggestions(
        arranged_file, 
        possible_file, 
        suggest_file, 
        index_file, 
        metal_file,
        max_suggestions=30
    )


def generate_final_suggestions(arranged_file, possible_file, suggest_file, 
                              index_file, metal_file, max_suggestions=30):
    """
    Select top suggestions from DFT-compatible configurations.
    
    Compares arranged predictions with possible DFT configurations and
    selects the top matches.
    
    Args:
        arranged_file (str): Sorted GPR predictions
        possible_file (str): DFT-compatible configurations
        suggest_file (str): Output file for suggested surface features
        index_file (str): Metal index file
        metal_file (str): Output file for suggested metal sequences
        max_suggestions (int): Maximum number of suggestions (default: 30)
    """
    with open(arranged_file, 'r') as t1, open(possible_file, 'r') as t2:
        arranged_lines = t1.readlines()
        possible_lines = t2.readlines()
    
    row_nums = []
    suggestions_count = 0
    
    print(f"  Selecting top {max_suggestions} DFT-compatible configurations...")
    
    with open(suggest_file, 'w') as outfile:
        for line in arranged_lines:
            if line in possible_lines:
                outfile.write(line)
                row_nums.append(possible_lines.index(line))
                suggestions_count += 1
                if suggestions_count >= max_suggestions:
                    break
    
    print(f"   Selected {suggestions_count} suggestions")
    print(f"   Saved to: {suggest_file}")
    
    with open(index_file, 'r') as f:
        index_to_metal = f.readlines()
    
    with open(metal_file, 'w') as fout:
        for num in row_nums:
            fout.write(index_to_metal[num])
    
    print(f"   Metal sequences saved to: {metal_file}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Usage:
        python mygaussian.py [batch_number]
    
    Example:
        python mygaussian.py 15
    """
    # Parse command line arguments
    if len(sys.argv) > 1:
        batch_number = int(sys.argv[1])
    else:
        batch_number = 15  # Default to final batch
        print(f"No batch number specified, using default: {batch_number}")
    
    print("="*60)
    print("Gaussian Process Regression for Active Learning")
    print("="*60)
    
    # File paths (modify these if your files have different names)
    o_data_file = 'DFT_O_all.csv'
    oh_data_file = 'DFT_OH_all.csv'
    
    if not os.path.exists(o_data_file):
        print(f"\nERROR: {o_data_file} not found")
        print("Please ensure your DFT data files are in the current directory")
        sys.exit(1)
    
    if not os.path.exists(oh_data_file):
        print(f"\nERROR: {oh_data_file} not found")
        print("Please ensure your DFT data files are in the current directory")
        sys.exit(1)
    
    try:
        # ====================================================================
        # Train Model 1: *O Adsorption
        # ====================================================================
        print("\n" + "="*60)
        print("Training Model 1: *O Adsorption")
        print("="*60)
        
        X_all, y_all = load_and_preprocess_data(o_data_file)
        x_train, x_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.25, random_state=42
        )
        
        print(f"  Training set: {len(x_train)} samples")
        print(f"  Test set: {len(x_test)} samples")
        
        model1 = create_gpr_model(n_features=15)
        print("  Training GPR model...")
        model1.fit(x_train, y_train)
        
        mae1 = evaluate_model(model1, x_test, y_test)
        print(f"   Model 1 MAE: {mae1:.4f} eV")
        
        # ====================================================================
        # Train Model 2: *OH Adsorption
        # ====================================================================
        print("\n" + "="*60)
        print("Training Model 2: *OH Adsorption")
        print("="*60)
        
        X_all2, y_all2 = load_and_preprocess_data(oh_data_file)
        x_train2, x_test2, y_train2, y_test2 = train_test_split(
            X_all2, y_all2, test_size=0.25, random_state=42
        )
        
        print(f"  Training set: {len(x_train2)} samples")
        print(f"  Test set: {len(x_test2)} samples")
        
        model2 = create_gpr_model(n_features=15)
        print("  Training GPR model...")
        model2.fit(x_train2, y_train2)
        
        mae2 = evaluate_model(model2, x_test2, y_test2)
        print(f"   Model 2 MAE: {mae2:.4f} eV")
        
        # ====================================================================
        # Generate Batch Suggestions
        # ====================================================================
        print("\n" + "="*60)
        print(f"Batch {batch_number} Generation")
        print("="*60)
        
        generate_batch_suggestions(model1, model2, batch_number)
        
        print("\n" + "="*60)
        print(" Active Learning Cycle Complete!")
        print("="*60)
        print(f"\nNext steps:")
        print(f"  1. Review batch{batch_number+1}_metal.csv for suggested structures")
        print(f"  2. Perform DFT calculations for selected configurations")
        print(f"  3. Add results to DFT_O_all.csv and DFT_OH_all.csv")
        print(f"  4. Run next iteration: python mygaussian.py {batch_number+1}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
